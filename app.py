import os
import cv2
import boto3
import uuid
import numpy as np
import logging
import shutil
import insightface
from sklearn.cluster import DBSCAN
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

# Configuration
TEMP_DIR = "/tmp"
VIDEO_FILENAME = "input_video.mp4"
FRAMES_DIR = os.path.join(TEMP_DIR, "frames")
RESULTS_DIR = os.path.join(TEMP_DIR, "results")

def lambda_handler(event, context):
    try:
        # Re-init Directories
        if os.path.exists(FRAMES_DIR): shutil.rmtree(FRAMES_DIR)
        if os.path.exists(RESULTS_DIR): shutil.rmtree(RESULTS_DIR)
        os.makedirs(FRAMES_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        bucket = event.get('bucket')
        key = event.get('key')
        if not bucket or not key: raise ValueError("Missing bucket/key")

        logger.info(f"Processing: s3://{bucket}/{key}")
        
        # 1. Download
        local_video_path = os.path.join(TEMP_DIR, VIDEO_FILENAME)
        logger.info("Downloading video...")
        s3_client.download_file(bucket, key, local_video_path)
        
        # 2. Init Models
        # Ensure model packs are available
        app = insightface.app.FaceAnalysis(name='buffalo_l', root=os.environ.get('INSIGHTFACE_HOME', '/tmp/.insightface'))
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 3. Analyze Video Metadata
        vid_cap = cv2.VideoCapture(local_video_path)
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        vid_cap.release()
        
        logger.info(f"Duration: {duration:.2f}s, FPS: {fps}, Frames: {total_frames}")

        all_faces = []

        # --- TIERED LOGIC ---
        
        if duration < 300: 
            # < 5 Minutes: SHORT VIDEO STRATEGY
            # Linear Scan (Pass 1 only). No seeking. Fast and simple.
            logger.info("Short Video Strategy (< 5 min): Linear Scan 1fps.")
            
            vid_cap = cv2.VideoCapture(local_video_path)
            frame_interval = int(fps) # 1 frame per second
            if frame_interval < 1: frame_interval = 1
            
            frame_count = 0
            processed_count = 0
            
            while vid_cap.isOpened():
                ret, frame = vid_cap.read()
                if not ret: break
                
                # Report progress
                if frame_count % (frame_interval * 30) == 0 and frame_count > 0:
                     logger.info(f"Progress: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")

                if frame_count % frame_interval == 0:
                    processed_count += 1
                    faces = app.get(frame)
                    for face in faces:
                        if face.det_score < 0.60: continue 
                        
                        # Score
                        yaw, pitch, roll = face.pose
                        pose_score = 100 - (abs(yaw)*1.2 + abs(pitch) + abs(roll)/2)
                        if pose_score < 0: pose_score = 0
                        
                        box = face.bbox
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        area = w * h
                        frame_area = frame.shape[0] * frame.shape[1]
                        size_ratio = area / frame_area
                        final_score = (face.det_score * 40) + (pose_score * 0.4) + (size_ratio * 100 * 0.2)
                        
                        frame_id = f"{frame_count}_{uuid.uuid4().hex[:8]}"
                        file_path = os.path.join(FRAMES_DIR, f"{frame_id}.jpg")
                        cv2.imwrite(file_path, frame)
                        
                        all_faces.append({
                            'embedding': face.embedding,
                            'score': final_score,
                            'file_path': file_path,
                            'frame_sec': frame_count / fps
                        })
                frame_count += 1
            vid_cap.release()
            logger.info(f"Linear Scan Complete. Processed {processed_count} frames.")

        else:
            # > 5 Minutes: LONG VIDEO STRATEGY
            # Smart 2-Pass to avoid reading empty segments.
            logger.info("Long Video Strategy (> 5 min): Smart 2-Pass Optimization.")
            
            # PASS 1: Coarse
            max_checks = 100
            step_frames = int(fps * 5) # Default 5s
            if total_frames / step_frames > max_checks:
                step_frames = int(total_frames / max_checks)
            
            coarse_indices = list(range(0, total_frames, step_frames))
            logger.info(f"[Pass 1] Checking {len(coarse_indices)} frames (Step: {step_frames})...")
            
            roi_timestamps = []
            vid_cap = cv2.VideoCapture(local_video_path)
            for i, idx in enumerate(coarse_indices):
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = vid_cap.read()
                if not ret: break
                
                if len(app.get(frame)) > 0:
                    roi_timestamps.append(idx / fps)
                
                if i % 20 == 0: logger.info(f"[Pass 1] Progress: {i}/{len(coarse_indices)}...")
            vid_cap.release()
            
            if not roi_timestamps:
                 return {"status": "success", "processed_key": key, "faces_found": 0}

            # PASS 2: Refine
            # Limit total refinement work
            MAX_REFINE_CHECKS = 150
            ranges = []
            roi_timestamps.sort()
            margin = 1.0
            
            curr_start = max(0, roi_timestamps[0] - margin)
            curr_end = roi_timestamps[0] + margin
            
            for t in roi_timestamps[1:]:
                start = max(0, t - margin)
                end = t + margin
                if start <= curr_end: curr_end = max(curr_end, end)
                else:
                    ranges.append((curr_start, curr_end))
                    curr_start = start
                    curr_end = end
            ranges.append((curr_start, curr_end))
            
            total_duration = sum([end - start for start, end in ranges])
            est_total_frames = total_duration * fps
            
            refine_step = int(est_total_frames / MAX_REFINE_CHECKS)
            if refine_step < 1: refine_step = 1
            # If dense step is too small (<5), force at least 5 frames gap to be efficient?
            # User said "10 frames", lets stick to reasonably sparse.
            if refine_step < 5: refine_step = 5
            
            num_segments = len(ranges)
            logger.info(f"[Pass 2] {num_segments} Segments. Est Frames: {est_total_frames:.0f}. Step: {refine_step} (every {refine_step/fps:.2f}s). Target Checks: ~{int(est_total_frames/refine_step)}")
            
            vid_cap = cv2.VideoCapture(local_video_path)
            for i, (start, end) in enumerate(ranges):
                if i % 5 == 0: logger.info(f"[Pass 2] Segment {i+1}/{num_segments}...")
                
                current_f = int(start * fps)
                end_f = int(end * fps)
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, current_f)
                
                while current_f <= end_f:
                    ret, frame = vid_cap.read()
                    if not ret: break
                    
                    faces = app.get(frame)
                    for face in faces:
                        if face.det_score < 0.60: continue
                        
                        yaw, pitch, roll = face.pose
                        pose_score = 100 - (abs(yaw)*1.2 + abs(pitch) + abs(roll)/2)
                        if pose_score < 0: pose_score = 0
                        
                        box = face.bbox
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        area = w * h
                        frame_area = frame.shape[0] * frame.shape[1]
                        size_ratio = area / frame_area
                        final_score = (face.det_score * 40) + (pose_score * 0.4) + (size_ratio * 100 * 0.2)
                        
                        frame_id = f"{current_f}_{uuid.uuid4().hex[:8]}"
                        file_path = os.path.join(FRAMES_DIR, f"{frame_id}.jpg")
                        cv2.imwrite(file_path, frame)
                        
                        all_faces.append({
                            'embedding': face.embedding,
                            'score': final_score,
                            'file_path': file_path,
                            'frame_sec': current_f / fps
                        })
                    
                    skip = refine_step - 1
                    if skip > 0:
                        current_f += skip
                        for _ in range(skip): vid_cap.grab()
                    current_f += 1
            vid_cap.release()

        # --- CLUSTERING ---
        if not all_faces:
             return {"status": "success", "processed_key": key, "faces_found": 0}

        logger.info(f"Clustering {len(all_faces)} candidates...")
        embeddings = [f['embedding'] for f in all_faces]
        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings)
            
            clustering = DBSCAN(eps=0.65, min_samples=3, metric='euclidean').fit(embeddings)
            labels = clustering.labels_
            
            unique_people = {} 
            for idx, label in enumerate(labels):
                if label == -1: continue 
                face_data = all_faces[idx]
                if label not in unique_people:
                    unique_people[label] = face_data
                else:
                    if face_data['score'] > unique_people[label]['score']:
                        unique_people[label] = face_data
            
            results = []
            base_name = os.path.splitext(os.path.basename(key))[0]
            
            for label, best_face in unique_people.items():
                src_path = best_face['file_path']
                dst_key = f"processed/{base_name}/person_{label}.jpg"
                
                img = cv2.imread(src_path)
                if img is not None:
                    h, w = img.shape[:2]
                    target_w, target_h = (1920, 1080) if w >= h else (1080, 1920)
                    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(src_path, resized)
                    s3_client.upload_file(src_path, bucket, dst_key)
                    
                    results.append({
                        "person_id": int(label),
                        "s3_key": dst_key,
                        "timestamp": best_face['frame_sec'],
                        "score": best_face['score'],
                        "resolution": f"{target_w}x{target_h}"
                    })
            
            logger.info(f"Done. Found {len(results)} unique people.")
            return {
                "status": "success", 
                "people_found": len(results), 
                "results": sorted(results, key=lambda x: x['person_id'])
            }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise e
