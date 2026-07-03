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
        # PINNED: onnxruntime-gpu 1.16.x dlopens libcudnn.so.8. Unpinned,
        # pip now resolves nvidia-cudnn-cu11 to cuDNN 9 (libcudnn.so.9) and
        # every inference silently falls back to CPUExecutionProvider —
        # ~10x slower scans on an idle, still-billed T4.
        "nvidia-cudnn-cu11==8.9.5.29",
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


def cluster_identities(embeddings):
    """
    Group L2-normalized face embeddings (one per sampled face) into unique people.

    Distances are euclidean on unit vectors, where d = sqrt(2 - 2*cos_sim), so
    EPS 1.0 == cosine similarity 0.5 — the standard identity threshold for
    ArcFace-family embeddings (Immich ships cosine distance 0.5 as its default).
    The old eps=0.65 demanded cos_sim >= 0.79 between samples, which split the
    same person across pose/lighting/blur changes into multiple "talents".

    Three passes:
      1. Core DBSCAN (min_samples=3) finds well-supported people.
      2. Noise points get re-assigned to the nearest person centroid within
         EPS, so borderline samples of a found person don't vanish.
      3. Remaining noise is clustered among itself (min_samples=2), so a person
         on screen only briefly (2+ samples) still becomes a person instead of
         being deleted. True singletons (one blink, one passer-by, one bad
         detection) stay dropped.

    Returns an int label per embedding; -1 = dropped.
    """
    import numpy as np
    from sklearn.cluster import DBSCAN

    EPS = 1.0  # euclidean on unit vectors == cosine similarity 0.5

    labels = DBSCAN(eps=EPS, min_samples=3, metric='euclidean').fit(embeddings).labels_

    # Pass 2: rescue noise samples that clearly belong to a found person.
    cluster_ids = sorted(set(labels) - {-1})
    if cluster_ids:
        centroids = []
        for cid in cluster_ids:
            c = embeddings[labels == cid].mean(axis=0)
            centroids.append(c / np.linalg.norm(c))
        centroids = np.array(centroids)
        for i in np.where(labels == -1)[0]:
            dists = np.linalg.norm(centroids - embeddings[i], axis=1)
            nearest = int(dists.argmin())
            if dists[nearest] <= EPS:
                labels[i] = cluster_ids[nearest]

    # Pass 3: promote brief appearances (2+ mutually-close leftover samples).
    leftover = np.where(labels == -1)[0]
    if len(leftover) >= 2:
        sub_labels = DBSCAN(eps=EPS, min_samples=2, metric='euclidean').fit(embeddings[leftover]).labels_
        next_id = (max(cluster_ids) + 1) if cluster_ids else 0
        for sub_id in sorted(set(sub_labels) - {-1}):
            for i in leftover[sub_labels == sub_id]:
                labels[i] = next_id
            next_id += 1

    return labels

@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("aws-s3-credentials")],
    volumes={"/root/.insightface": model_cache},
)
def extract_frames(bucket: str, main_folder: str, account_id: str, content_id: str, video_key: str, transcript_key: str = None, custom_metadata: dict = None, faster_using_zoom_transcript: bool = False) -> dict:
    import cv2
    import boto3
    import uuid
    import time
    import numpy as np
    import insightface
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

        # 1. Transcript first (cheap S3 get) — fast mode needs the speaker
        #    turns before we decide how to read the video.
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
            except Exception as e:
                print(f"WARNING: transcript parse failed for {transcript_key}: {e}")

        def get_context(ts):
            for e in transcript_data:
                if e['start'] <= ts <= e['end']: return e['speaker'], e['text']
            for e in transcript_data:
                if abs(e['start'] - ts) < 2.0: return e['speaker'], e['text']
            return None, None

        # Zoom-style VTTs label every caption "Speaker Name: text". When the
        # caller sets faster_using_zoom_transcript, those names ARE the talent
        # list: probe a few frames inside each speaker's longest turns instead
        # of scanning the whole video (minutes -> seconds on long webinars).
        named_speakers = {}
        if faster_using_zoom_transcript:
            for e in transcript_data:
                if e['speaker']:
                    named_speakers.setdefault(e['speaker'], []).append((e['start'], e['end']))
        use_fast = bool(named_speakers)
        mode = "fast_zoom_transcript" if use_fast else "full_scan"

        # 2. Open the video. Fast mode streams the presigned URL (a handful
        #    of seeks beats downloading a 1h webinar); the full scan downloads
        #    first for reliable sequential decode.
        video_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': video_key}, ExpiresIn=3600)
        local_video = os.path.join(temp_dir, "video.mp4")

        def read_meta(path):
            c = cv2.VideoCapture(path)
            f = c.get(cv2.CAP_PROP_FPS)
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            c.release()
            return f, n, w, h

        if use_fast:
            active_path = video_url
            fps, total_frames, v_w, v_h = read_meta(active_path)
            if fps <= 0 or total_frames <= 0:
                print("Stream metadata unreadable - falling back to full scan")
                use_fast = False
                mode = "full_scan"
        if not use_fast:
            print(f"Downloading video: s3://{bucket}/{video_key}")
            s3.download_file(bucket, video_key, local_video)
            print("Download complete.")
            active_path = local_video
            fps, total_frames, v_w, v_h = read_meta(active_path)
        duration = total_frames / fps if fps > 0 else 0

        # 3. Init Face AI (GPU) - Optimized for memory efficiency
        # NOTE: insightface does NOT raise when CUDA is unavailable — onnxruntime
        # silently applies CPUExecutionProvider. Check the provider list for the
        # truth instead of trusting the absence of an exception.
        import onnxruntime
        gpu_inference = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        if not gpu_inference:
            print("*** CRITICAL: CUDAExecutionProvider unavailable — inference will run on CPU (slow) ***")
        face_app = insightface.app.FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=0 if gpu_inference else -1, det_size=(320, 320))  # Smaller detection size for memory efficiency
        print(f"Face detection provider: {'GPU (CUDA)' if gpu_inference else 'CPU FALLBACK'}")

        # 4. Find talent candidates
        all_faces = []
        fast_talents = []
        if use_fast:
            print(f"FAST MODE: probing turns for {len(named_speakers)} transcript speaker(s)")
            cap = cv2.VideoCapture(active_path)
            for speaker, turns in named_speakers.items():
                # Probe midpoints of the speaker's longest turns; the best
                # face across those probes becomes their talent frame.
                probes = sorted(turns, key=lambda t: t[1] - t[0], reverse=True)[:6]
                best = None
                for t_start, t_end in probes:
                    ts = (t_start + t_end) / 2.0
                    f_idx = max(0, min(int(ts * fps), total_frames - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if not ret: continue
                    for face in face_app.get(frame):
                        if face.det_score < 0.6: continue
                        box, yaw, pitch, roll = face.bbox, *face.pose
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        size_ratio = area / (frame.shape[0] * frame.shape[1])
                        pose_score = max(0, 100 - (abs(yaw) * 1.2 + abs(pitch) + abs(roll) / 2))
                        q_score = (face.det_score * 40) + (pose_score * 0.4) + (size_ratio * 100 * 0.2)
                        if best is None or q_score > best['score']:
                            best = {'frame': frame.copy(), 'score': q_score, 'timestamp': ts}
                if best is not None:
                    _, context = get_context(best['timestamp'])
                    fast_talents.append({'name': speaker, 'context_text': context, **best})
                else:
                    print(f"FAST MODE: no face found for speaker '{speaker}' (camera off?)")
            cap.release()
        else:
            # FULL SCAN - Memory optimized: only store face data, not frame images
            stride_sec = 1.0 if duration < 600 else (2.0 if duration < 1800 else 5.0)
            stride_frames = max(1, int(fps * stride_sec))
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
        if use_fast:
            for pid, talent in enumerate(fast_talents):
                frame = talent['frame']
                out_path = os.path.join(temp_dir, f"p_{pid}.jpg")
                cv2.imwrite(out_path, cv2.resize(frame, (1920, 1080) if frame.shape[1] >= frame.shape[0] else (1080, 1920)))
                s_key = f"{out_prefix}/person_{pid}.jpg"
                results_client.upload_file(out_path, bucket, s_key, ExtraArgs={'ContentType': 'image/jpeg'})
                talent_results.append({
                    "person_id": pid, "name": talent['name'],
                    "context_text": talent['context_text'], "s3_url": f"https://{bucket}.s3.amazonaws.com/{s_key}",
                    "timestamp": round(talent['timestamp'], 2), "score": round(float(talent['score']), 2)
                })
        elif all_faces:
            embeddings = normalize(np.array([f['embedding'] for f in all_faces]))
            labels = cluster_identities(embeddings)
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
            "status": "success", "account_id": account_id, "content_id": content_id, "mode": mode,
            "custom_metadata": custom_metadata, "processing_metrics": {"duration_seconds": proc_time, "estimated_cost_usd": round(proc_time * 0.000416, 4), "gpu_type": "NVIDIA T4", "gpu_inference": gpu_inference},
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
        request.get("custom_metadata"),
        request.get("faster_using_zoom_transcript", False)
    )


@app.local_entrypoint()
def main(bucket: str, main_folder: str, account_id: str, content_id: str, video_key: str, transcript: str = None):
    print(extract_frames.remote(bucket, main_folder, account_id, content_id, video_key, transcript))
