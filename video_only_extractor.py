import modal
import os
import shutil
import tempfile
import uuid
from datetime import datetime

# Use debian_slim with build-time brute-force symlinking for GPU libs
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "insightface==0.7.3",
        "onnxruntime-gpu==1.16.3",
        "nvidia-cuda-runtime-cu11",
        "nvidia-cudnn-cu11",
        "nvidia-cublas-cu11",
        "nvidia-cuda-nvrtc-cu11",
        "opencv-python-headless==4.9.0.80",
        "boto3==1.34.0",
        "scikit-learn==1.4.0",
        "numpy==1.26.3",
        "fastapi",
        "httpx",
        "webvtt-py==0.4.6",
        "requests",
        "m3u8",
    )
    .run_commands(
        "find /usr/local/lib/python3.11/site-packages/nvidia -name '*.so*' -exec ln -sf {} /usr/lib/ \\;"
    )
)

app = modal.App("video-only-extractor", image=image)
model_cache = modal.Volume.from_name("insightface-models", create_if_missing=True)

@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("aws-s3-credentials")],
    volumes={"/root/.insightface": model_cache},
)
def extract_video_frames(video_url: str, transcript_url: str = None, metadata: dict = None) -> dict:
    """
    Extract talent frames from video with job-based organization.

    Args:
        video_url: Direct URL to video file
        transcript_url: Direct URL to VTT transcript (optional)
        metadata: Additional metadata for the job

    Returns:
        Dict with extracted frames info and S3 URLs
    """
    import cv2
    import boto3
    import time
    import numpy as np
    import insightface
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    import requests

    start_perf = time.perf_counter()

    # Generate job identifiers
    job_id = str(uuid.uuid4())
    date_folder = datetime.now().strftime("%Y-%m-%d")

    # S3 setup
    s3_client = boto3.client('s3')
    bucket_name = "logie-users"

    try:
        temp_dir = tempfile.mkdtemp()

        # 1. Download video (handle HLS streams for short videos)
        print(f"Downloading video from: {video_url}")

        if video_url.endswith('.m3u8') or 'hls' in video_url.lower():
            try:
                print("Detected HLS stream - attempting to download and convert segments...")
                video_path = download_hls_to_mp4(video_url, temp_dir)
            except Exception as hls_error:
                print(f"HLS download failed: {hls_error}")
                print("Attempting direct download of HLS URL as fallback...")

                # Fallback: try downloading the HLS URL directly (sometimes works)
                try:
                    hls_response = requests.get(video_url, stream=True, timeout=30)
                    hls_response.raise_for_status()

                    video_path = os.path.join(temp_dir, "video.mp4")
                    with open(video_path, 'wb') as f:
                        for chunk in hls_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print("Direct HLS download succeeded as fallback")
                except Exception as direct_error:
                    print(f"Direct HLS download also failed: {direct_error}")
                    raise ValueError(f"Both HLS parsing and direct download failed. HLS Error: {hls_error}, Direct Error: {direct_error}")
        else:
            # Direct MP4 download
            video_response = requests.get(video_url, stream=True, timeout=30)
            video_response.raise_for_status()

            video_path = os.path.join(temp_dir, "video.mp4")
            with open(video_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Video download complete.")

        # 2. Download transcript if provided
        transcript_data = []
        if transcript_url:
            print(f"Downloading transcript from: {transcript_url}")
            transcript_response = requests.get(transcript_url)
            transcript_response.raise_for_status()

            import webvtt
            transcript_content = transcript_response.text

            with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False, encoding='utf-8') as tf:
                tf.write(transcript_content)
                vtt_path = tf.name

            for caption in webvtt.read(vtt_path):
                txt = caption.text.strip().replace('\n', ' ')
                speaker = None
                if ':' in txt:
                    parts = txt.split(':', 1)
                    if len(parts[0]) < 40:
                        speaker = parts[0].strip()
                        txt = parts[1].strip()
                transcript_data.append({
                    'start': caption.start_in_seconds,
                    'end': caption.end_in_seconds,
                    'speaker': speaker,
                    'text': txt
                })
            os.remove(vtt_path)
            print("Transcript download complete.")

        # 3. Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"Video metadata: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps, {v_w}x{v_h}")

        # 4. Initialize face detection (memory optimized)
        try:
            face_app = insightface.app.FaceAnalysis(name='buffalo_l')
            face_app.prepare(ctx_id=0, det_size=(320, 320))
            print("Using GPU for face detection")
        except Exception as e:
            print(f"GPU failed, falling back to CPU: {e}")
            face_app = insightface.app.FaceAnalysis(name='buffalo_l')
            face_app.prepare(ctx_id=-1, det_size=(320, 320))

        # 5. Scan video for faces (memory efficient)
        stride_sec = 1.0 if duration < 600 else (2.0 if duration < 1800 else 5.0)
        stride_frames = max(1, int(fps * stride_sec))
        all_faces = []

        print(f"Scanning every {stride_sec}s...")
        cap = cv2.VideoCapture(video_path)
        processed_frames = 0

        for f_idx in range(0, total_frames, stride_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break

            faces = face_app.get(frame)
            for face in faces:
                if face.det_score < 0.6: continue
                box, yaw, pitch, roll = face.bbox, *face.pose
                area = (box[2] - box[0]) * (box[3] - box[1])
                size_ratio = area / (frame.shape[0] * frame.shape[1])
                pose_score = max(0, 100 - (abs(yaw) * 1.2 + abs(pitch) + abs(roll) / 2))
                q_score = (face.det_score * 40) + (pose_score * 0.4) + (size_ratio * 100 * 0.2)

                all_faces.append({
                    'embedding': face.embedding,
                    'score': q_score,
                    'frame_idx': f_idx,
                    'timestamp': f_idx / fps,
                    'bbox': box
                })

            processed_frames += 1
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames, found {len(all_faces)} faces so far...")

        cap.release()
        print(f"Scan complete. Found {len(all_faces)} faces in {processed_frames} frames.")

        # Memory cleanup
        import gc
        gc.collect()

        # 6. Process transcript context
        def get_context(ts):
            for e in transcript_data:
                if e['start'] <= ts <= e['end']: return e['speaker'], e['text']
            for e in transcript_data:
                if abs(e['start'] - ts) < 2.0: return e['speaker'], e['text']
            return None, None

        # 7. Generate representative frames
        rep_indices = set([0, min(29, total_frames-1), max(0, total_frames-31), max(0, total_frames-2)])
        if total_frames > 10:
            for p in np.linspace(30, total_frames-31, 10).astype(int)[1:-1]: rep_indices.add(int(p))
        while len(rep_indices) < 10 and len(rep_indices) < total_frames: rep_indices.add(len(rep_indices))
        sorted_rep = sorted(list(rep_indices))[:10]

        # 8. Upload results to job-based folder structure
        base_path = f"extracted-frames/{date_folder}/{job_id}"
        talent_results = []

        # Upload representative frames
        rep_results = []
        cap = cv2.VideoCapture(video_path)
        for i, f_idx in enumerate(sorted_rep):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue

            frame_filename = f"representative_frame_{i}.jpg"
            s3_key = f"{base_path}/{frame_filename}"

            # Save and upload
            frame_path = os.path.join(temp_dir, frame_filename)
            cv2.imwrite(frame_path, cv2.resize(frame, (1920, 1080) if frame.shape[1] >= frame.shape[0] else (1080, 1920)))
            s3_client.upload_file(frame_path, bucket_name, s3_key, ExtraArgs={'ContentType': 'image/jpeg'})

            rep_results.append({
                "frame_index": int(f_idx),
                "filename": frame_filename,
                "s3_key": s3_key,
                "s3_url": f"https://{bucket_name}.s3.amazonaws.com/{s3_key}",
                "timestamp": round(f_idx / fps, 2)
            })
        cap.release()

        # Process and upload talent frames
        if all_faces:
            embeddings = normalize(np.array([f['embedding'] for f in all_faces]))
            labels = DBSCAN(eps=0.65, min_samples=3).fit(embeddings).labels_
            unique = {}

            for i, l in enumerate(labels):
                if l != -1 and (l not in unique or all_faces[i]['score'] > unique[l]['score']):
                    unique[l] = all_faces[i]

            for person_id, face_data in unique.items():
                speaker, context = get_context(face_data['timestamp'])

                # Re-extract frame
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, face_data['frame_idx'])
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame_filename = f"person_{person_id}.jpg"
                    s3_key = f"{base_path}/{frame_filename}"

                    # Save and upload
                    frame_path = os.path.join(temp_dir, frame_filename)
                    cv2.imwrite(frame_path, cv2.resize(frame, (1920, 1080) if frame.shape[1] >= frame.shape[0] else (1080, 1920)))
                    s3_client.upload_file(frame_path, bucket_name, s3_key, ExtraArgs={'ContentType': 'image/jpeg'})

                    talent_results.append({
                        "person_id": int(person_id),
                        "filename": frame_filename,
                        "s3_key": s3_key,
                        "s3_url": f"https://{bucket_name}.s3.amazonaws.com/{s3_key}",
                        "name": speaker if speaker else f"Person {person_id}",
                        "context_text": context,
                        "timestamp": round(face_data['timestamp'], 2),
                        "score": round(float(face_data['score']), 2)
                    })

        # Calculate processing metrics
        proc_time = round(time.perf_counter() - start_perf, 2)
        gpu_cost = round(proc_time * 0.000416, 4)  # T4 GPU cost per second

        result = {
            "status": "success",
            "job_id": job_id,
            "date_folder": date_folder,
            "base_path": base_path,
            "metadata": metadata or {},
            "processing_metrics": {
                "duration_seconds": proc_time,
                "estimated_cost_usd": gpu_cost,
                "gpu_type": "NVIDIA T4"
            },
            "video_metadata": {
                "duration_seconds": round(duration, 2),
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "resolution": f"{v_w}x{v_h}",
                "source_url": video_url
            },
            "transcript_metadata": {
                "source_url": transcript_url,
                "entries_count": len(transcript_data)
            } if transcript_url else None,
            "talent_count": len(talent_results),
            "talent_frames": sorted(talent_results, key=lambda x: x['person_id']),
            "representative_frames": rep_results
        }

        # Send webhook
        try:
            import httpx
            httpx.post(
                "https://hook.us1.make.com/qb8jajua119emykshhxdkl7wrbrct4cr",
                json=result,
                timeout=10.0
            )
        except Exception as e:
            print(f"Webhook failed: {e}")

        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def download_hls_to_mp4(hls_url: str, temp_dir: str) -> str:
    """
    Download HLS stream and convert to MP4 for short videos (10-30 seconds).

    Args:
        hls_url: URL to the HLS .m3u8 playlist
        temp_dir: Temporary directory to store files

    Returns:
        Path to the converted MP4 file
    """
    import m3u8
    import urllib.parse

    output_path = os.path.join(temp_dir, "video.mp4")

    try:
        print(f"Starting HLS download from: {hls_url}")

        # Download and parse HLS playlist
        print("Downloading HLS playlist...")
        playlist_response = requests.get(hls_url, timeout=10)
        playlist_response.raise_for_status()
        print(f"Playlist downloaded, {len(playlist_response.text)} characters")

        # Parse the playlist
        print("Parsing HLS playlist...")
        playlist = m3u8.loads(playlist_response.text)

        if not playlist.segments:
            print("No segments found in playlist, checking if it's a variant playlist...")
            # Check if it's a variant playlist (multiple quality options)
            if playlist.playlists:
                print(f"Found {len(playlist.playlists)} quality variants, using first one")
                # Use the first (usually highest quality) variant
                variant_url = urllib.parse.urljoin(hls_url, playlist.playlists[0].uri)
                print(f"Downloading variant playlist: {variant_url}")
                variant_response = requests.get(variant_url, timeout=10)
                variant_response.raise_for_status()
                playlist = m3u8.loads(variant_response.text)
                hls_url = variant_url  # Update base URL for segments

        if not playlist.segments:
            raise ValueError(f"No segments found in HLS playlist. Content: {playlist_response.text[:500]}")

        print(f"Found {len(playlist.segments)} HLS segments to download")

        # Download all segments
        segment_files = []
        base_url = urllib.parse.urljoin(hls_url, '.')

        for i, segment in enumerate(playlist.segments):
            segment_url = urllib.parse.urljoin(base_url, segment.uri)
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.ts")

            try:
                # Download segment
                segment_response = requests.get(segment_url, timeout=15)
                segment_response.raise_for_status()

                with open(segment_path, 'wb') as f:
                    f.write(segment_response.content)

                segment_files.append(segment_path)
                print(f"Downloaded segment {i+1}/{len(playlist.segments)} ({len(segment_response.content)} bytes)")

            except Exception as seg_e:
                print(f"Failed to download segment {i+1}: {segment_url} - {seg_e}")
                # Continue with other segments if possible
                continue

        if not segment_files:
            raise ValueError("Failed to download any HLS segments")

        # Concatenate all segments into MP4
        print(f"Concatenating {len(segment_files)} segments into MP4...")
        with open(output_path, 'wb') as outfile:
            for segment_file in segment_files:
                with open(segment_file, 'rb') as infile:
                    outfile.write(infile.read())

        # Verify the output file
        if os.path.getsize(output_path) == 0:
            raise ValueError("Created MP4 file is empty")

        print(f"Successfully created MP4 ({os.path.getsize(output_path)} bytes) from {len(segment_files)} HLS segments")

        # Clean up segment files
        for segment_file in segment_files:
            try:
                os.remove(segment_file)
            except OSError:
                pass

        return output_path

    except Exception as e:
        print(f"HLS download failed: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to download HLS stream: {str(e)}")


@app.function()
@modal.fastapi_endpoint(method="POST")
def process_video_job(request: dict) -> dict:
    """
    Simplified API endpoint for video-only processing.

    Supports multiple payload formats:

    1. Direct URLs:
    {
        "video_url": "https://example.com/video.mp4",
        "transcript_url": "https://example.com/transcript.vtt",
        "metadata": {"job_name": "My Video"}
    }

    2. Amazon Live data:
    {
        "amazon_data": {
            "broadcast_id": "02c4ee7e633246e384019e387bbf6db4",
            "hls_url": "https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/...",
            "closed_captions": "en,https://m.media-amazon.com/images/S/vse-vms-closed-captions-artifact-us-east-1-prod/...",
            "video_preview_assets": [{"url": "https://...", "type": "default"}],
            ...
        },
        "metadata": {"shop_id": "influencer-7feb78c5", "broadcast_title": "..."}
    }
    """
    metadata = request.get("metadata", {})

    # Handle Amazon Live data format
    if "amazon_data" in request:
        amazon_data = request["amazon_data"]

        # Extract video URL - prioritize HLS for short videos (10-30 seconds)
        video_url = None

        # For short Amazon Live videos, HLS is preferred (can be downloaded quickly)
        if "hls_url" in amazon_data and amazon_data["hls_url"]:
            video_url = amazon_data["hls_url"]
            print(f"Using HLS stream for short video: {video_url}")

        # Fallback to MP4 preview if HLS not available
        if not video_url and "video_preview_assets" in amazon_data and amazon_data["video_preview_assets"]:
            print("Falling back to video preview assets")
            # Use the first/default MP4 preview
            for asset in amazon_data["video_preview_assets"]:
                if asset.get("mimeType") == "video/mp4" or asset.get("type") == "default":
                    video_url = asset["url"]
                    break
            if not video_url and amazon_data["video_preview_assets"]:
                video_url = amazon_data["video_preview_assets"][0]["url"]

        # Final fallback: construct full video URL from broadcast_id
        if not video_url:
            broadcast_id = amazon_data.get("broadcast_id")
            if broadcast_id:
                video_url = f"https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/{broadcast_id}/default.jobtemplate.mp4"
                print(f"Final fallback: constructed full video URL from broadcast_id: {video_url}")

        # Extract transcript URL from closed_captions
        transcript_url = None
        if "closed_captions" in amazon_data and amazon_data["closed_captions"]:
            # Format: "en,https://url.vtt" or just "https://url.vtt"
            captions = amazon_data["closed_captions"]
            if "," in captions:
                parts = captions.split(",", 1)
                transcript_url = parts[1] if len(parts) > 1 else parts[0]
            else:
                transcript_url = captions

        # Merge Amazon metadata
        if amazon_data:
            metadata.update({
                "broadcast_id": amazon_data.get("broadcast_id"),
                "shop_id": amazon_data.get("shop_id"),
                "broadcast_title": amazon_data.get("broadcast_title"),
                "aci_content_id": amazon_data.get("aci_content_id"),
                "reference_id": amazon_data.get("reference_id"),
                "synopsis": amazon_data.get("synopsis"),
                "formatted_duration": amazon_data.get("formatted_duration"),
            })

    else:
        # Direct URL format
        video_url = request.get("video_url")
        transcript_url = request.get("transcript_url")

    # Validate required parameters
    if not video_url:
        raise ValueError("Missing video URL. Provide either 'video_url' or 'amazon_data' with video assets.")

    print(f"Processing video: {video_url}")
    if transcript_url:
        print(f"Using transcript: {transcript_url}")

    return extract_video_frames.remote(video_url, transcript_url, metadata)


@app.function()
@modal.fastapi_endpoint(method="POST")
def test_hls_url(request: dict) -> dict:
    """
    Test endpoint to debug HLS URL accessibility.

    Payload: {"hls_url": "https://example.com/playlist.m3u8"}
    """
    hls_url = request.get("hls_url")
    if not hls_url:
        return {"error": "Missing hls_url parameter"}

    try:
        # Test playlist accessibility
        response = requests.get(hls_url, timeout=10)
        response.raise_for_status()

        # Try to parse as M3U8
        import m3u8
        playlist = m3u8.loads(response.text)

        result = {
            "playlist_accessible": True,
            "playlist_size": len(response.text),
            "is_variant_playlist": len(playlist.playlists) > 0,
            "segment_count": len(playlist.segments) if playlist.segments else 0,
            "variant_count": len(playlist.playlists) if playlist.playlists else 0
        }

        # Test first segment if available
        if playlist.segments:
            import urllib.parse
            base_url = urllib.parse.urljoin(hls_url, '.')
            first_segment_url = urllib.parse.urljoin(base_url, playlist.segments[0].uri)

            try:
                segment_response = requests.head(first_segment_url, timeout=5)
                result["first_segment_accessible"] = segment_response.status_code == 200
                result["first_segment_url"] = first_segment_url
            except Exception as seg_e:
                result["first_segment_error"] = str(seg_e)
                result["first_segment_url"] = first_segment_url

        return result

    except Exception as e:
        return {
            "error": str(e),
            "playlist_accessible": False,
            "hls_url": hls_url
        }


@app.local_entrypoint()
def main(video_url: str, transcript_url: str = None):
    """Local testing entrypoint"""
    import json
    result = extract_video_frames.remote(video_url, transcript_url)
    print(json.dumps(result, indent=2))
