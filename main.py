import modal
import os
import shutil
import tempfile
import uuid
import time
import sys
from datetime import datetime

# -----------------------------------------------------------------------------
# IMAGE SETUP: NVIDIA Devel + Build Tools (The configuration that worked)
# -----------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "libgl1", 
        "libglib2.0-0", 
        "ffmpeg", 
        "git", 
        "wget",
        "build-essential", # Required for InsightFace
        "clang",           # Required for InsightFace
        "python3-dev" 
    )
    .pip_install("cython", "numpy==1.26.3", "setuptools", "wheel")
    .pip_install(
        "insightface==0.7.3",
        "onnxruntime-gpu==1.16.3",
        "opencv-python-headless==4.9.0.80",
        "boto3==1.34.0",
        "scikit-learn==1.4.0",
        "fastapi",
        "httpx",
        "requests",
        "m3u8",
        "ffmpeg-python",
        "webvtt-py==0.4.6" # Required for transcripts
    )
    .run_commands(
        "pip uninstall -y onnxruntime || true", 
        "echo 'Cleaned CPU libs'"
    )
)

# You can name this whatever you want to keep your URL consistent
app = modal.App("video-extractor-final", image=image)
model_cache = modal.Volume.from_name("insightface-models", create_if_missing=True)

# -----------------------------------------------------------------------------
# UNIFIED FUNCTION (GPU + WEB ENDPOINT)
# -----------------------------------------------------------------------------
@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("aws-s3-credentials")],
    volumes={"/root/.insightface": model_cache},
    container_idle_timeout=120,
    allow_concurrent_inputs=1,
)
@modal.fastapi_endpoint(method="POST")
def process(request: dict) -> dict:
    """
    Full Logic Restored: Transcripts, Representative Frames, and Archival.
    """
    import cv2
    import boto3
    import numpy as np
    import insightface
    import onnxruntime
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    import requests
    import webvtt

    start_perf = time.perf_counter()

    # 1. HARDWARE CHECK
    if os.system("nvidia-smi > /dev/null 2>&1") != 0:
        print("❌ CRITICAL: No GPU attached!")
    
    if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        raise RuntimeError("❌ ONNX Runtime is missing CUDA. Aborting.")

    # 2. PARSE INPUTS (Amazon or Direct)
    metadata = request.get("metadata", {})
    video_url = request.get("video_url")
    transcript_url = request.get("transcript_url")
    
    if "amazon_data" in request:
        amz = request["amazon_data"]
        metadata.update({
            "aci_content_id": amz.get("aci_content_id"),
            "broadcast_id": amz.get("broadcast_id"),
            "shop_id": amz.get("shop_id")
        })
        # Prioritize HLS
        video_url = amz.get("hls_url")
        if not video_url and amz.get("video_preview_assets"):
            video_url = amz["video_preview_assets"][0].get("url")
        
        # Extract Transcript URL
        if not transcript_url and amz.get("closed_captions"):
            caps = amz["closed_captions"]
            transcript_url = caps.split(",")[1] if "," in caps else caps

    if not video_url:
        return {"error": "No video_url found"}

    # 3. SETUP & DOWNLOAD
    job_id = str(uuid.uuid4())
    date_folder = datetime.now().strftime("%Y-%m-%d")
    s3_client = boto3.client('s3')
    bucket_name = "logie-users"
    
    temp_dir = tempfile.mkdtemp()
    try:
        # --- A. Download Video ---
        print(f"⬇️ Downloading Video: {video_url}")
        video_path = os.path.join(temp_dir, "video.mp4")
        
        if "m3u8" in video_url or video_url.endswith(".m3u8"):
            import ffmpeg
            try:
                (
                    ffmpeg
                    .input(video_url)
                    .output(video_path, vcodec='copy', acodec='copy', avoid_negative_ts='make_zero')
                    .run(quiet=True, overwrite_output=True)
                )
            except Exception:
                # Fallback re-encode
                (
                    ffmpeg
                    .input(video_url)
                    .output(video_path, preset='ultrafast')
                    .run(quiet=True, overwrite_output=True)
                )
        else:
            r = requests.get(video_url, stream=True, timeout=60)
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # --- B. Download Transcript ---
        transcript_data = []
        if transcript_url:
            print(f"⬇️ Downloading Transcript: {transcript_url}")
            try:
                tr = requests.get(transcript_url)
                vtt_path = os.path.join(temp_dir, "subs.vtt")
                with open(vtt_path, 'w', encoding='utf-8') as f:
                    f.write(tr.text)
                
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
            except Exception as e:
                print(f"⚠️ Transcript failed: {e}")

        # --- C. Video Metadata ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"📊 Stats: {duration:.1f}s | {fps:.1f} FPS | {v_w}x{v_h}")

        # --- D. Initialize GPU Inference ---
        print("🚀 Initializing InsightFace on GPU...")
        face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        # --- E. Extract Representative Frames (Start, Middle, End) ---
        rep_indices = set([0, min(29, total_frames-1), max(0, total_frames-31), max(0, total_frames-2)])
        if total_frames > 10:
            for p in np.linspace(30, total_frames-31, 10).astype(int)[1:-1]: rep_indices.add(int(p))
        sorted_rep = sorted(list(rep_indices))[:10]
        
        base_path = f"extracted-frames/{date_folder}/{job_id}"
        rep_results = []
        
        cap = cv2.VideoCapture(video_path)
        for i, f_idx in enumerate(sorted_rep):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            fname = f"representative_frame_{i}.jpg"
            key = f"{base_path}/{fname}"
            local_p = os.path.join(temp_dir, fname)
            
            # Resize logic from original
            h, w = frame.shape[:2]
            target = (1920, 1080) if w >= h else (1080, 1920)
            cv2.imwrite(local_p, cv2.resize(frame, target))
            
            s3_client.upload_file(local_p, bucket_name, key, ExtraArgs={'ContentType': 'image/jpeg'})
            rep_results.append({
                "frame_index": f_idx,
                "filename": fname,
                "s3_key": key,
                "s3_url": f"https://{bucket_name}.s3.amazonaws.com/{key}",
                "timestamp": round(f_idx / fps, 2)
            })
        cap.release()

        # --- F. Scan for Faces ---
        all_faces = []
        stride = max(1, int(fps * (1.0 if duration < 600 else 2.0)))
        cap = cv2.VideoCapture(video_path)
        
        f_idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            
            faces = face_app.get(frame)
            for face in faces:
                if face.det_score > 0.6:
                    all_faces.append({
                        'embedding': face.embedding,
                        'score': float(face.det_score),
                        'frame_idx': f_idx,
                        'timestamp': f_idx / fps
                    })
            
            f_idx += stride
            if f_idx >= total_frames: break
        cap.release()

        # --- G. Cluster & Upload Talent Frames ---
        talent_results = []
        
        def get_context(ts):
            for e in transcript_data:
                if e['start'] <= ts <= e['end']: return e['speaker'], e['text']
            for e in transcript_data:
                if abs(e['start'] - ts) < 2.0: return e['speaker'], e['text']
            return None, None

        if all_faces:
            print(f"🧠 Clustering {len(all_faces)} faces...")
            feats = normalize(np.array([f['embedding'] for f in all_faces]))
            labels = DBSCAN(eps=0.65, min_samples=3).fit(feats).labels_
            
            unique = {}
            for i, l in enumerate(labels):
                if l != -1 and (l not in unique or all_faces[i]['score'] > unique[l]['score']):
                    unique[l] = all_faces[i]
            
            cap = cv2.VideoCapture(video_path)
            for pid, data in unique.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame_idx'])
                ret, frame = cap.read()
                if ret:
                    speaker, context = get_context(data['timestamp'])
                    
                    fname = f"person_{pid}.jpg"
                    key = f"{base_path}/{fname}"
                    local_p = os.path.join(temp_dir, fname)
                    
                    h, w = frame.shape[:2]
                    target = (1920, 1080) if w >= h else (1080, 1920)
                    cv2.imwrite(local_p, cv2.resize(frame, target))
                    
                    s3_client.upload_file(local_p, bucket_name, key, ExtraArgs={'ContentType': 'image/jpeg'})
                    
                    talent_results.append({
                        "person_id": int(pid),
                        "filename": fname,
                        "s3_key": key,
                        "s3_url": f"https://{bucket_name}.s3.amazonaws.com/{key}",
                        "name": speaker if speaker else f"Person {pid}",
                        "context_text": context,
                        "timestamp": round(data['timestamp'], 2),
                        "score": round(data['score'], 2)
                    })
            cap.release()

        # --- H. Archive Full Video ---
        content_id = metadata.get('aci_content_id') or metadata.get('content_id') or job_id
        vid_filename = f"{content_id}.mp4"
        vid_key = f"amazon-shorts/{date_folder}/{vid_filename}"
        
        print(f"💾 Archiving video to {vid_key}")
        try:
            s3_client.upload_file(video_path, bucket_name, vid_key, ExtraArgs={
                'ContentType': 'video/mp4',
                'Metadata': {'job_id': job_id}
            })
            video_s3_url = f"https://{bucket_name}.s3.amazonaws.com/{vid_key}"
        except Exception as e:
            print(f"Archive failed: {e}")
            video_s3_url = None

        # --- I. Final Metrics & Text ---
        proc_time = round(time.perf_counter() - start_perf, 2)
        gpu_cost = round(proc_time * 0.000416, 4)
        
        transcription_text = ""
        if transcript_data:
            parts = []
            for e in transcript_data:
                if e.get('text'):
                    ts = f"[{e['start']:.1f}s] "
                    spk = f"{e.get('speaker')}: " if e.get('speaker') else ""
                    parts.append(f"{ts}{spk}{e['text']}")
            transcription_text = " ".join(parts)
            if len(transcription_text) > 1000:
                transcription_text = transcription_text[:997] + "..."

        # --- J. Construct Original Payload ---
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
                "source_url": video_url,
                "archived_s3_key": vid_key,
                "archived_s3_url": video_s3_url
            },
            "transcript_metadata": {
                "source_url": transcript_url,
                "entries_count": len(transcript_data)
            } if transcript_url else None,
            "transcription": transcription_text,
            "talent_count": len(talent_results),
            "talent_frames": sorted(talent_results, key=lambda x: x['person_id']),
            "representative_frames": rep_results
        }

        # --- K. Webhook ---
        try:
            print("🚀 Sending Webhook...")
            requests.post(
                "https://hook.us1.make.com/qb8jajua119emykshhxdkl7wrbrct4cr",
                json=result,
                timeout=5.0
            )
        except Exception:
            pass

        return result

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)