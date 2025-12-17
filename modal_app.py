"""
Modal GPU Implementation - Video Talent Frame Extractor
Runs on T4 GPU for ~10-20x faster processing than Lambda CPU.
"""
import modal
import os

# --- MODAL SETUP ---
# Define container image with GPU support
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV deps
    .pip_install(
        "insightface==0.7.3",
        "onnxruntime-gpu==1.16.3",
        "opencv-python-headless==4.9.0.80",
        "boto3==1.34.0",
        "scikit-learn==1.4.0",
        "numpy==1.26.3",
    )
)

app = modal.App("video-talent-extractor", image=image)

# Create a Volume to cache InsightFace models (avoids re-download on each run)
model_cache = modal.Volume.from_name("insightface-models", create_if_missing=True)

# --- CORE FUNCTION ---
@app.function(
    gpu="T4",  # NVIDIA T4 GPU (~$0.000416/sec)
    timeout=600,  # 10 min max
    secrets=[modal.Secret.from_name("aws-s3-credentials")],  # Your S3 creds
    volumes={"/root/.insightface": model_cache},  # Cache models
)
def extract_frames(bucket: str, key: str) -> dict:
    """
    Main processing function. Runs on GPU.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path to video)
    
    Returns:
        dict with status, people_found, and results array
    """
    import cv2
    import boto3
    import uuid
    import numpy as np
    import insightface
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    import tempfile
    import shutil

    # Init S3 (credentials come from Modal Secret)
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )

    # Setup temp dirs
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir)

    try:
        # Download video
        video_path = os.path.join(temp_dir, "input.mp4")
        print(f"Downloading s3://{bucket}/{key}...")
        s3.download_file(bucket, key, video_path)

        # Init InsightFace (uses GPU automatically via onnxruntime-gpu)
        face_app = insightface.app.FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 = GPU

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        print(f"Video: {duration:.1f}s, {total_frames} frames @ {fps:.1f} fps")

        all_faces = []

        # --- TIERED LOGIC (same as Lambda version) ---
        if duration < 300:
            # Short video: Linear scan
            print("Short video mode: Linear scan...")
            cap = cv2.VideoCapture(video_path)
            interval = int(fps)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    faces = face_app.get(frame)
                    for face in faces:
                        if face.det_score < 0.6:
                            continue
                        score = _calculate_score(face, frame)
                        path = os.path.join(frames_dir, f"{frame_idx}_{uuid.uuid4().hex[:8]}.jpg")
                        cv2.imwrite(path, frame)
                        all_faces.append({
                            'embedding': face.embedding,
                            'score': score,
                            'path': path,
                            'timestamp': frame_idx / fps
                        })
                frame_idx += 1
            cap.release()
        else:
            # Long video: 2-Pass
            print("Long video mode: 2-Pass scan...")
            # Pass 1: Coarse
            cap = cv2.VideoCapture(video_path)
            step = max(int(fps * 5), int(total_frames / 100))
            roi_times = []

            for idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                if len(face_app.get(frame)) > 0:
                    roi_times.append(idx / fps)
            cap.release()

            if not roi_times:
                return {"status": "success", "people_found": 0, "results": []}

            # Build ranges
            roi_times.sort()
            ranges = []
            margin = 1.0
            start, end = max(0, roi_times[0] - margin), roi_times[0] + margin
            for t in roi_times[1:]:
                if t - margin <= end:
                    end = max(end, t + margin)
                else:
                    ranges.append((start, end))
                    start, end = max(0, t - margin), t + margin
            ranges.append((start, end))

            # Pass 2: Refine
            cap = cv2.VideoCapture(video_path)
            total_dur = sum(e - s for s, e in ranges)
            refine_step = max(5, int(total_dur * fps / 150))

            for s, e in ranges:
                sf, ef = int(s * fps), int(e * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
                curr = sf
                while curr <= ef:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    faces = face_app.get(frame)
                    for face in faces:
                        if face.det_score < 0.6:
                            continue
                        score = _calculate_score(face, frame)
                        path = os.path.join(frames_dir, f"{curr}_{uuid.uuid4().hex[:8]}.jpg")
                        cv2.imwrite(path, frame)
                        all_faces.append({
                            'embedding': face.embedding,
                            'score': score,
                            'path': path,
                            'timestamp': curr / fps
                        })
                    for _ in range(refine_step - 1):
                        cap.grab()
                    curr += refine_step
            cap.release()

        print(f"Found {len(all_faces)} face candidates.")

        if not all_faces:
            return {"status": "success", "people_found": 0, "results": []}

        # --- CLUSTERING ---
        embeddings = normalize(np.array([f['embedding'] for f in all_faces]))
        labels = DBSCAN(eps=0.65, min_samples=3).fit(embeddings).labels_

        unique = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            if label not in unique or all_faces[idx]['score'] > unique[label]['score']:
                unique[label] = all_faces[idx]

        # --- UPLOAD RESULTS ---
        base_name = os.path.splitext(os.path.basename(key))[0]
        results = []

        for label, data in unique.items():
            img = cv2.imread(data['path'])
            h, w = img.shape[:2]
            target = (1920, 1080) if w >= h else (1080, 1920)
            resized = cv2.resize(img, target)

            # Save locally then upload
            out_path = os.path.join(temp_dir, f"person_{label}.jpg")
            cv2.imwrite(out_path, resized)

            s3_key = f"processed/{base_name}/person_{label}.jpg"
            s3.upload_file(out_path, bucket, s3_key)

            results.append({
                "person_id": int(label),
                "s3_key": s3_key,
                "s3_url": f"https://{bucket}.s3.amazonaws.com/{s3_key}",
                "timestamp": data['timestamp'],
                "score": data['score']
            })

        print(f"Done! {len(results)} unique people.")
        return {
            "status": "success",
            "people_found": len(results),
            "results": sorted(results, key=lambda x: x['person_id'])
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
@modal.web_endpoint(method="POST")
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
