"""
Modal GPU Implementation - Video Talent Frame Extractor
Runs on T4 GPU for ~10-20x faster processing than Lambda CPU.
"""
import modal
import os

# --- MODAL SETUP ---
# Define container image with GPU support
# We use a base image that's good for GPU or install the specific nvidia libraries ORT needs
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
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
    })
)

app = modal.App("video-talent-extractor", image=image)

# Create a Volume to cache InsightFace models
model_cache = modal.Volume.from_name("insightface-models", create_if_missing=True)

# --- CORE FUNCTION ---
@app.function(
    gpu="T4",
    timeout=900,  # Increased to 15m for safety on very long videos
    secrets=[modal.Secret.from_name("aws-s3-credentials")],
    volumes={"/root/.insightface": model_cache},
)
def extract_frames(bucket: str, key: str) -> dict:
    """
    Highly optimized Single-Pass GPU processing with cost tracking.
    """
    import cv2
    import boto3
    import uuid
    import time
    import numpy as np
    import insightface
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    import tempfile
    import shutil

    start_perf = time.perf_counter()

    # --- FORCE GPU LIBS VISIBILITY ---
    # Symlink pip-installed nvidia libs to system path so ONNX can find them
    import glob
    for lib_path in glob.glob("/usr/local/lib/python3.11/site-packages/nvidia/*/lib/*.so*"):
        try:
            target = f"/usr/lib/x86_64-linux-gnu/{os.path.basename(lib_path)}"
            if not os.path.exists(target):
                os.symlink(lib_path, target)
        except Exception as e:
            print(f"Symlink failed for {lib_path}: {e}")

    # Init S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )

    try:
        # --- STREAMING URL ---
        video_url = s3.generate_presigned_url(
            'get_object', 
            Params={'Bucket': bucket, 'Key': key}, 
            ExpiresIn=3600
        )

        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)

        # Init InsightFace (GPU enabled)
        face_app = insightface.app.FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Get video info
        cap = cv2.VideoCapture(video_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"Video: {duration:.1f}s, {total_frames} frames @ {fps:.1f} fps")

        # --- OPTIMIZED SINGLE-PASS SCAN ---
        # Dynamic stride: Faster for long videos, thorough for short ones.
        if duration < 600:
            stride_sec = 1.0  # 1 FPS for videos < 10m
        elif duration < 1800:
            stride_sec = 2.0  # 1 frame every 2s for 10-30m
        else:
            stride_sec = 5.0  # 1 frame every 5s for > 30m
        
        stride_frames = max(1, int(fps * stride_sec))
        all_faces = []

        print(f"Single-Pass Scan: Every {stride_sec}s (stride: {stride_frames})...")
        cap = cv2.VideoCapture(video_url)
        for f_idx in range(0, total_frames, stride_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            
            faces = face_app.get(frame)
            for face in faces:
                if face.det_score < 0.6: continue
                score = _calculate_score(face, frame)
                # Keep if high quality or first face found
                path = os.path.join(frames_dir, f"{f_idx}_{uuid.uuid4().hex[:8]}.jpg")
                cv2.imwrite(path, frame)
                all_faces.append({'embedding': face.embedding, 'score': score, 'path': path, 'timestamp': f_idx / fps})
        cap.release()

        # --- REPRESENTATIVE FRAMES (Force Exactly 10) ---
        rep_indices = set()
        rep_indices.add(0)
        rep_indices.add(min(29, total_frames - 1))
        rep_indices.add(max(0, total_frames - 31))
        rep_indices.add(max(0, total_frames - 2)) 
        
        if total_frames > 10:
            dist_pts = np.linspace(30, total_frames - 31, 10).astype(int)[1:-1]
            for p in dist_pts: rep_indices.add(int(p))
        
        idx = 1
        while len(rep_indices) < 10 and idx < total_frames:
            rep_indices.add(idx)
            idx += 1
        
        sorted_rep = sorted(list(rep_indices))[:10]

        # S3 Pathing
        account_id, content_id = "unknown", "unknown"
        parts = key.split('/')
        if len(parts) >= 4 and parts[0] == 'content':
            account_id, content_id = parts[1], parts[2]
            output_bucket = "logie-users"
            output_prefix = f"content/{account_id}/{content_id}/extraction-talent-frames"
        else:
            output_bucket = bucket
            output_prefix = f"processed/{os.path.splitext(os.path.basename(key))[0]}"

        s3_extra = {'ContentType': 'image/jpeg'}

        # Upload Representative
        rep_results = []
        cap = cv2.VideoCapture(video_url)
        for i, f_idx in enumerate(sorted_rep):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_idx - 1))
                ret, frame = cap.read()
                if not ret: continue
            
            h, w = frame.shape[:2]
            target = (1920, 1080) if w >= h else (1080, 1920)
            resized = cv2.resize(frame, target)
            out_path = os.path.join(temp_dir, f"rep_{i}.jpg")
            cv2.imwrite(out_path, resized)
            s_key = f"{output_prefix}/representative_frame_{i}.jpg"
            s3.upload_file(out_path, output_bucket, s_key, ExtraArgs=s3_extra)
            rep_results.append({
                "frame_index": f_idx, 
                "s3_url": f"https://{output_bucket}.s3.amazonaws.com/{s_key}", 
                "timestamp": round(f_idx / fps, 2)
            })
        cap.release()

        # Cluster and Upload Talent
        talent_results = []
        if all_faces:
            embeddings = normalize(np.array([f['embedding'] for f in all_faces]))
            labels = DBSCAN(eps=0.65, min_samples=3).fit(embeddings).labels_
            unique = {}
            for i, label in enumerate(labels):
                if label == -1: continue
                if label not in unique or all_faces[i]['score'] > unique[label]['score']:
                    unique[label] = all_faces[i]
            
            for label, data in unique.items():
                img = cv2.imread(data['path'])
                h, w = img.shape[:2]
                target = (1920, 1080) if w >= h else (1080, 1920)
                resized = cv2.resize(img, target)
                out_path = os.path.join(temp_dir, f"p_{label}.jpg")
                cv2.imwrite(out_path, resized)
                s_key = f"{output_prefix}/person_{label}.jpg"
                s3.upload_file(out_path, output_bucket, s_key, ExtraArgs=s3_extra)
                talent_results.append({
                    "person_id": int(label),
                    "s3_url": f"https://{output_bucket}.s3.amazonaws.com/{s_key}",
                    "timestamp": round(data['timestamp'], 2),
                    "score": round(float(data['score']), 2)
                })

        # --- FINAL PERFORMANCE METRICS ---
        end_perf = time.perf_counter()
        proc_time = round(end_perf - start_perf, 2)
        # Modal T4 pricing is ~$0.000416/sec ($1.50/hr)
        est_cost = round(proc_time * 0.000416, 4)

        final_result = {
            "status": "success",
            "account_id": account_id,
            "content_id": content_id,
            "processing_metrics": {
                "duration_seconds": proc_time,
                "estimated_cost_usd": est_cost,
                "gpu_type": "NVIDIA T4"
            },
            "video_metadata": {
                "duration_seconds": round(duration, 2),
                "total_frames": total_frames,
                "fps": round(fps, 2)
            },
            "talent_count": len(talent_results),
            "talent_frames": sorted(talent_results, key=lambda x: x['person_id']),
            "representative_frames": rep_results
        }

        # Webhook
        webhook_url = "https://hook.us1.make.com/qb8jajua119emykshhxdkl7wrbrct4cr"
        import httpx
        try:
            print(f"Sending webhook ({proc_time}s, ${est_cost})...")
            httpx.post(webhook_url, json=final_result, timeout=10.0)
        except Exception as e: print(f"Webhook error: {e}")
        
        return final_result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        # Send failure webhook if possible
        try:
            import httpx
            webhook_url = "https://hook.us1.make.com/qb8jajua119emykshhxdkl7wrbrct4cr"
            httpx.post(webhook_url, json={"status": "error", "error": str(e), "key": key}, timeout=5.0)
        except: pass
        raise e

    finally:
        t_dir = locals().get('temp_dir')
        if t_dir and os.path.exists(t_dir):
            shutil.rmtree(t_dir, ignore_errors=True)


def _calculate_score(face, frame):
    """Calculate quality score for a face."""
    yaw, pitch, roll = face.pose
    pose_score = max(0, 100 - (abs(yaw) * 1.2 + abs(pitch) + abs(roll) / 2))
    box = face.bbox
    area = (box[2] - box[0]) * (box[3] - box[1])
    size_ratio = area / (frame.shape[0] * frame.shape[1])
    return (face.det_score * 40) + (pose_score * 0.4) + (size_ratio * 100 * 0.2)


# --- WEB ENDPOINT ---
@app.function()
@modal.fastapi_endpoint(method="POST")
def process_video(request: dict) -> dict:
    """
    HTTPS Endpoint to trigger processing.
    
    POST Body:
        {"bucket": "my-bucket", "key": "videos/my-video.mp4"}
    
    Returns:
        Processing results with S3 URLs
    """
    bucket = request.get("bucket")
    key = request.get("key")

    if not bucket or not key:
        return {"error": "Missing 'bucket' or 'key' in request"}

    # Run the GPU function
    result = extract_frames.remote(bucket, key)
    return result


# --- CLI ENTRYPOINT ---
@app.local_entrypoint()
def main(bucket: str, key: str):
    """
    CLI: modal run modal_app.py --bucket my-bucket --key video.mp4
    """
    result = extract_frames.remote(bucket, key)
    print(result)
