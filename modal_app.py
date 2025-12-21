import modal
import os
import shutil
import tempfile

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
    )
    .run_commands(
        "find /usr/local/lib/python3.11/site-packages/nvidia -name '*.so*' -exec ln -sf {} /usr/lib/ \\;"
    )
)

app = modal.App("video-talent-extractor", image=image)
model_cache = modal.Volume.from_name("insightface-models", create_if_missing=True)

@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("aws-s3-credentials")],
    volumes={"/root/.insightface": model_cache},
)
def extract_frames(bucket: str, main_folder: str, account_id: str, content_id: str, video_key: str, transcript_key: str = None, custom_metadata: dict = None) -> dict:
    import cv2
    import boto3
    import uuid
    import time
    import numpy as np
    import insightface
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    import glob

    # Validate required parameters
    if not bucket or not main_folder or not account_id or not content_id or not video_key:
        raise ValueError("Missing required parameters: bucket, main_folder, account_id, content_id, and video_key must all be provided and non-empty")

    start_perf = time.perf_counter()

    # Init S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )

    try:
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)

        # 1. Download video first (always, for reliable metadata extraction)
        print(f"Downloading video: s3://{bucket}/{video_key}")
        local_video = os.path.join(temp_dir, "video.mp4")
        s3.download_file(bucket, video_key, local_video)
        print("Download complete.")

        # 2. Get metadata from local file
        cap = cv2.VideoCapture(local_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        active_path = local_video

        # 3. Init Face AI (GPU) - Optimized for memory efficiency
        try:
            face_app = insightface.app.FaceAnalysis(name='buffalo_l')
            face_app.prepare(ctx_id=0, det_size=(320, 320))  # Smaller detection size for memory efficiency
            print("Using GPU for face detection")
        except Exception as e:
            print(f"GPU failed, falling back to CPU: {e}")
            face_app = insightface.app.FaceAnalysis(name='buffalo_l')
            face_app.prepare(ctx_id=-1, det_size=(320, 320))  # Smaller size for CPU too

        # 4. SCAN - Memory optimized: only store face data, not frame images
        stride_sec = 1.0 if duration < 600 else (2.0 if duration < 1800 else 5.0)
        stride_frames = max(1, int(fps * stride_sec))
        all_faces = []
        print(f"Scanning every {stride_sec}s...")
        cap = cv2.VideoCapture(active_path)
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

                # Store frame data in memory only - don't save images yet
                all_faces.append({
                    'embedding': face.embedding,
                    'score': q_score,
                    'frame_idx': f_idx,
                    'timestamp': f_idx / fps,
                    'bbox': box
                })

            processed_frames += 1
            # Progress logging every 100 frames to avoid spam
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames, found {len(all_faces)} faces so far...")

        cap.release()
        print(f"Scan complete. Found {len(all_faces)} faces in {processed_frames} frames.")

        # Memory cleanup - force garbage collection
        import gc
        gc.collect()
        
        # 5. Transcript
        transcript_data = []
        if transcript_key:
            try:
                import webvtt
                vtt_resp = s3.get_object(Bucket=bucket, Key=transcript_key)
                vtt_content = vtt_resp['Body'].read().decode('utf-8')
                with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False, encoding='utf-8') as tf:
                    tf.write(vtt_content)
                    vtt_path = tf.name
                for caption in webvtt.read(vtt_path):
                    txt = caption.text.strip().replace('\n', ' ')
                    speaker = None
                    if ':' in txt:
                        parts = txt.split(':', 1)
                        if len(parts[0]) < 40:
                            speaker = parts[0].strip()
                            txt = parts[1].strip()
                    transcript_data.append({'start': caption.start_in_seconds, 'end': caption.end_in_seconds, 'speaker': speaker, 'text': txt})
                os.remove(vtt_path)
            except: pass

        def get_context(ts):
            for e in transcript_data:
                if e['start'] <= ts <= e['end']: return e['speaker'], e['text']
            for e in transcript_data:
                if abs(e['start'] - ts) < 2.0: return e['speaker'], e['text']
            return None, None

        # 6. Rep Frames (10)
        rep_indices = set([0, min(29, total_frames-1), max(0, total_frames-31), max(0, total_frames-2)])
        if total_frames > 10:
            for p in np.linspace(30, total_frames-31, 10).astype(int)[1:-1]: rep_indices.add(int(p))
        while len(rep_indices) < 10 and len(rep_indices) < total_frames: rep_indices.add(len(rep_indices))
        sorted_rep = sorted(list(rep_indices))[:10]

        # 7. Upload
        results_client = boto3.client('s3')
        out_prefix = f"{main_folder}/{account_id}/{content_id}/extraction-talent-frames"
        
        rep_results = []
        cap = cv2.VideoCapture(active_path)  # Reopen for representative frames
        for i, f_idx in enumerate(sorted_rep):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            out_path = os.path.join(temp_dir, f"rep_{i}.jpg")
            cv2.imwrite(out_path, cv2.resize(frame, (1920, 1080) if frame.shape[1] >= frame.shape[0] else (1080, 1920)))
            s_key = f"{out_prefix}/representative_frame_{i}.jpg"
            results_client.upload_file(out_path, bucket, s_key, ExtraArgs={'ContentType': 'image/jpeg'})
            rep_results.append({"frame_index": f_idx, "s3_url": f"https://{bucket}.s3.amazonaws.com/{s_key}", "timestamp": round(f_idx / fps, 2)})
        cap.release()

        talent_results = []
        if all_faces:
            embeddings = normalize(np.array([f['embedding'] for f in all_faces]))
            labels = DBSCAN(eps=0.65, min_samples=3).fit(embeddings).labels_
            unique = {}
            for i, l in enumerate(labels):
                if l != -1 and (l not in unique or all_faces[i]['score'] > unique[l]['score']): unique[l] = all_faces[i]
            for l, data in unique.items():
                speaker, context = get_context(data['timestamp'])

                # Re-extract the frame from video using stored frame index
                cap = cv2.VideoCapture(active_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame_idx'])
                ret, frame = cap.read()
                cap.release()

                if ret:
                    out_path = os.path.join(temp_dir, f"p_{l}.jpg")
                    cv2.imwrite(out_path, cv2.resize(frame, (1920, 1080) if frame.shape[1] >= frame.shape[0] else (1080, 1920)))
                    s_key = f"{out_prefix}/person_{l}.jpg"
                    results_client.upload_file(out_path, bucket, s_key, ExtraArgs={'ContentType': 'image/jpeg'})
                    talent_results.append({
                        "person_id": int(l), "name": speaker if speaker else f"Person {l}",
                        "context_text": context, "s3_url": f"https://{bucket}.s3.amazonaws.com/{s_key}",
                        "timestamp": round(data['timestamp'], 2), "score": round(float(data['score']), 2)
                    })
        cap.release()

        # Final Metrics
        proc_time = round(time.perf_counter() - start_perf, 2)
        res = {
            "status": "success", "account_id": account_id, "content_id": content_id,
            "custom_metadata": custom_metadata, "processing_metrics": {"duration_seconds": proc_time, "estimated_cost_usd": round(proc_time * 0.000416, 4), "gpu_type": "NVIDIA T4"},
            "video_metadata": {"duration_seconds": round(duration, 2), "total_frames": total_frames, "fps": round(fps, 2), "resolution": f"{v_w}x{v_h}"},
            "talent_count": len(talent_results), "talent_frames": sorted(talent_results, key=lambda x: x['person_id']), "representative_frames": rep_results
        }
        try:
            import httpx
            httpx.post("https://hook.us1.make.com/qb8jajua119emykshhxdkl7wrbrct4cr", json=res, timeout=10.0)
        except: pass
        return res

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise e
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.function()
@modal.fastapi_endpoint(method="POST")
def process_video(request: dict) -> dict:
    # Validate required parameters
    required_params = ["bucket", "main_folder", "account_id", "content_id", "video_key"]
    for param in required_params:
        if param not in request or request[param] is None:
            raise ValueError(f"Missing required parameter: {param}")

    return extract_frames.remote(
        request["bucket"],
        request["main_folder"],
        request["account_id"],
        request["content_id"],
        request["video_key"],
        request.get("transcript_key"),
        request.get("custom_metadata")
    )


@app.local_entrypoint()
def main(bucket: str, main_folder: str, account_id: str, content_id: str, video_key: str, transcript: str = None):
    print(extract_frames.remote(bucket, main_folder, account_id, content_id, video_key, transcript))
