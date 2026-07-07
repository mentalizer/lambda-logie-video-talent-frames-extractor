import modal
import os
import shutil
import subprocess
import tempfile
import uuid
import time
import json
from datetime import datetime

# -----------------------------------------------------------------------------
# IMAGE SETUP: CUDA 11.8 + cuDNN 8 — the combo that works with onnxruntime-gpu
# 1.16 (InsightFace) AND ctranslate2 3.x (faster-whisper), so face detection and
# transcription share the same T4.
# -----------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "ffmpeg",
        "git",
        "wget",
        "build-essential",  # Required for InsightFace
        "clang",            # Required for InsightFace
        "python3-dev",
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
        "webvtt-py==0.4.6",  # Required for transcripts
    )
    .run_commands(
        "pip uninstall -y onnxruntime || true",
        "echo 'Cleaned CPU libs'",
    )
    # -------------------------------------------------------------------------
    # Whisper stack — appended AFTER the original layers so everything above
    # stays cached from previous builds. Pins match CUDA 11.8 / cuDNN 8.
    # pkg-config + FFmpeg dev headers: Modal's PyPI mirror serves `av` as an
    # sdist, so PyAV compiles from source against the system FFmpeg.
    # -------------------------------------------------------------------------
    .apt_install(
        "pkg-config",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libavfilter-dev",
        "libswscale-dev",
        "libswresample-dev",
    )
    .pip_install(
        "ctranslate2==3.24.0",
        "av==11.0.0",
        "tokenizers==0.15.2",
        "huggingface_hub==0.25.2",
    )
    # --no-deps: faster-whisper declares CPU `onnxruntime` as a dependency, which
    # would clobber onnxruntime-gpu. All of its real deps are pinned above.
    .pip_install("faster-whisper==0.10.1", extra_options="--no-deps")
)

# Keep the app + function names stable so the deployed URL never changes.
app = modal.App("video-extractor-final", image=image)
model_cache = modal.Volume.from_name("insightface-models", create_if_missing=True)
whisper_cache = modal.Volume.from_name("whisper-models", create_if_missing=True)

# faster-whisper model used for warm-up preload + transcription.
# NOTE: this image pins faster-whisper==0.10.1 / ctranslate2==3.24.0 (CUDA 11.8,
# to coexist with InsightFace's onnxruntime-gpu 1.16.3). That version CANNOT load
# large-v3-turbo (released later -> 500). distil-large-v3 (CT2) DOES load here and
# is ~2x faster than large-v3 on short clips (whisper 8.6s -> 4.6s measured), with
# near-identical English accuracy and fewer hallucinations.
# ENGLISH-ONLY — the transcript path auto-routes non-English audio to
# MULTILINGUAL_WHISPER_MODEL below, so this stays the fast English default.
DEFAULT_WHISPER_MODEL = "Systran/faster-distil-whisper-large-v3"
# Non-English fallback (distil is English-only) + a cheap multilingual model used
# only to detect the language before choosing the transcription model.
MULTILINGUAL_WHISPER_MODEL = "large-v3"
LANG_DETECT_MODEL = "base"


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


# -----------------------------------------------------------------------------
# CONTAINER-LEVEL MODEL SINGLETONS
# Loaded once per container, reused across warm requests — model init no longer
# costs 5-10s on every call.
# -----------------------------------------------------------------------------
_FACE_APP = None
_WHISPER_MODELS = {}    # name -> WhisperModel; several coexist (distil + large-v3 + detector)
_WHISPER_LOAD_S = None  # seconds the last NEW Whisper load took (warm/cold verification)
_COLD = True            # flipped False after this container serves its first real request


def _get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        import insightface
        print("🚀 Initializing InsightFace on GPU...")
        fa = insightface.app.FaceAnalysis(
            name='buffalo_l',
            # Skip genderage/landmark_2d_106 — landmark_3d_68 gives us head pose
            # for the quality score, recognition gives embeddings.
            allowed_modules=['detection', 'recognition', 'landmark_3d_68'],
            providers=['CUDAExecutionProvider'],
        )
        fa.prepare(ctx_id=0, det_size=(640, 640))
        det = fa.models.get('detection')
        if det is not None and 'CUDAExecutionProvider' not in det.session.get_providers():
            raise RuntimeError("❌ Face detector session fell back to CPU. Aborting.")
        _FACE_APP = fa
    return _FACE_APP


def _get_whisper(model_name):
    """Load + cache a faster-whisper model by name. Multiple models coexist
    (English distil + multilingual large-v3 + the language detector) so routing
    between them never evicts/reloads. All fit a 16GB T4 alongside InsightFace."""
    global _WHISPER_LOAD_S
    if model_name not in _WHISPER_MODELS:
        from faster_whisper import WhisperModel
        print(f"🎙️ Loading Whisper '{model_name}' on GPU...")
        _t = time.perf_counter()
        _WHISPER_MODELS[model_name] = WhisperModel(
            model_name,
            device="cuda",
            compute_type="int8_float16",  # T4-friendly: fast + fits easily in 16GB
            download_root="/root/.cache/whisper",
        )
        _WHISPER_LOAD_S = round(time.perf_counter() - _t, 2)
        print(f"🎙️ Whisper '{model_name}' loaded in {_WHISPER_LOAD_S}s")
    return _WHISPER_MODELS[model_name]


def _detect_language(audio_path):
    """Cheap multilingual language detection with the base model. faster-whisper
    computes info.language EAGERLY (before the lazy segment generator is
    consumed), so reading it here costs only the first-window encode
    (~0.3-0.5s), not a full transcription. Returns (lang_code, probability)."""
    try:
        model = _get_whisper(LANG_DETECT_MODEL)
        _segments, info = model.transcribe(audio_path, beam_size=1, language=None)
        return info.language, round(float(info.language_probability), 3)
    except Exception as e:
        print(f"⚠️ language detection failed ({e}); defaulting to en")
        return "en", 0.0


def _make_s3(bucket_override=None):
    """Env-driven S3/R2 client shared by the transcript-only fast path.
    OBJECT_STORAGE_ENDPOINT_URL present -> Cloudflare R2; absent -> AWS S3."""
    import boto3
    bucket_name = bucket_override or os.environ.get("OBJECT_STORAGE_BUCKET", "logie-users")
    endpoint_url = os.environ.get("OBJECT_STORAGE_ENDPOINT_URL")
    if endpoint_url:
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
    else:
        s3_client = boto3.client('s3')
    public_base = (
        os.environ.get("OBJECT_STORAGE_PUBLIC_BASE_URL")
        or f"https://{bucket_name}.s3.amazonaws.com"
    ).rstrip('/')
    return s3_client, bucket_name, public_base


def _transcribe_only(request, task_id="", cold=False):
    """Transcript-only fast path: download staged video -> ffmpeg 16kHz mono
    audio -> Whisper -> transcript. Skips the whole face/cluster/archive
    pipeline so the response returns as soon as Whisper finishes. This is what
    replaces the Groq + Transloadit round-trip and unblocks the LLM brief."""
    import requests
    from urllib.parse import quote

    start_perf = time.perf_counter()
    timings = {}
    whisper_model_name = request.get("whisper_model", DEFAULT_WHISPER_MODEL)
    language = request.get("language")

    bucket_override = request.get("bucket")
    location = (request.get("location") or "").strip().strip("/")
    source_filename = (request.get("filename") or "").strip().strip("/")
    video_url = request.get("video_url")
    if not source_filename and not video_url:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={
            "status": "error",
            "error": "transcript_only needs bucket/location/filename or video_url",
            "task_id": task_id, "cold": cold,
        })

    s3_client, bucket_name, public_base = _make_s3(bucket_override)

    # Tolerate a location that repeats the bucket name (copied from the dashboard).
    if location == bucket_name or location.startswith(bucket_name + "/"):
        location = location[len(bucket_name):].strip("/")
    source_key = (f"{location}/{source_filename}" if location else source_filename) if source_filename else None
    stem = os.path.splitext(source_filename)[0] if source_filename else "video"
    name_prefix = f"{stem}_" if source_filename else ""

    temp_dir = tempfile.mkdtemp()
    try:
        audio_path = os.path.join(temp_dir, "audio.wav")

        def _extract_audio(src):
            # -vn: drop video. reconnect flags make HTTP(S) inputs resilient to
            # transient drops while streaming from the presigned URL.
            return subprocess.run(
                ["ffmpeg", "-y",
                 "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                 "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path],
                capture_output=True,
            ).returncode

        # 1+2. Stream audio straight from the source — no full-video download to
        # disk. The transcript only needs audio, so we presign a GET URL (S3/R2)
        # and let ffmpeg pull + extract 16kHz mono in ONE overlapped pass. NB: the
        # separate talent-frames call still downloads the whole video (it needs the
        # pixels); these are independent invocations so nothing was shared anyway —
        # this just makes the transcript path lighter and faster.
        t0 = time.perf_counter()
        if source_key:
            try:
                source_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": source_key},
                    ExpiresIn=600,
                )
            except Exception as e:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=404, content={
                    "status": "error",
                    "error": f"Could not presign '{source_key}' in '{bucket_name}': {e}",
                    "task_id": task_id, "cold": cold,
                })
        else:
            source_url = video_url

        rc = _extract_audio(source_url)
        streamed_ok = rc == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024
        timings["stream_extract"] = round(time.perf_counter() - t0, 2)

        # Fallback: some MP4s (moov atom at the end) don't stream over HTTP —
        # download the whole file, then extract locally.
        if not streamed_ok:
            t0 = time.perf_counter()
            video_path = os.path.join(temp_dir, "video.mp4")
            if source_key:
                try:
                    s3_client.download_file(bucket_name, source_key, video_path)
                except Exception as e:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(status_code=404, content={
                        "status": "error",
                        "error": f"Could not fetch '{source_key}' from bucket '{bucket_name}': {e}",
                        "task_id": task_id, "cold": cold,
                    })
            else:
                r = requests.get(video_url, stream=True, timeout=60)
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            rc = _extract_audio(video_path)
            timings["download_fallback"] = round(time.perf_counter() - t0, 2)
            if rc != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) <= 1024:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=422, content={
                    "status": "error",
                    "error": "No usable audio track to transcribe",
                    "task_id": task_id, "cold": cold,
                })

        # 3. Route the model by language: distil (fast) for English, large-v3
        # (multilingual) for everything else. Skip detection when the caller
        # already knows the language (pass "language"), or disables it.
        english_model = whisper_model_name  # the request's preferred/English model
        multilingual_model = request.get("whisper_model_multilingual", MULTILINGUAL_WHISPER_MODEL)
        detected_lang = None
        detected_prob = None
        if language:
            lang = language
        elif request.get("detect_language", True):
            t0 = time.perf_counter()
            detected_lang, detected_prob = _detect_language(audio_path)
            timings["lang_detect"] = round(time.perf_counter() - t0, 2)
            lang = detected_lang
        else:
            lang = "en"

        if lang == "en":
            chosen_model, tr_language = english_model, "en"
        else:
            chosen_model, tr_language = multilingual_model, lang

        # 4. Transcribe with the chosen model. beam_size=1 (greedy): the LLM brief
        # needs semantic content, not verbatim precision, and greedy is markedly
        # faster. VAD on to skip silence and suppress silent-gap hallucinations.
        t0 = time.perf_counter()
        model = _get_whisper(chosen_model)
        segments, info = model.transcribe(
            audio_path, beam_size=1, vad_filter=True, language=tr_language,
        )
        entries = []
        for seg in segments:
            txt = seg.text.strip()
            if txt:
                entries.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "speaker": None,
                    "text": txt,
                })
        timings["whisper"] = round(time.perf_counter() - t0, 2)

        text = " ".join(e["text"] for e in entries).strip()
        whisper_info = {
            "model": chosen_model,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "detected_language": detected_lang,
            "detection_probability": detected_prob,
            "routed": "english" if lang == "en" else "multilingual",
        }

        # 4. Persist the full transcript.json next to the source (storage mode).
        transcript_s3_key = None
        transcript_s3_url = None
        if entries and source_filename:
            try:
                transcript_s3_key = (
                    f"{location}/{name_prefix}transcript.json" if location else f"{name_prefix}transcript.json"
                )
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=transcript_s3_key,
                    Body=json.dumps(
                        {"source": "whisper", "whisper": whisper_info, "entries": entries},
                        ensure_ascii=False,
                    ).encode("utf-8"),
                    ContentType='application/json',
                )
                transcript_s3_url = f"{public_base}/{quote(transcript_s3_key)}"
            except Exception as e:
                print(f"⚠️ Transcript upload failed: {e}")

        return {
            "status": "success",
            "mode": "transcript",
            "task_id": task_id,
            "cold": cold,
            "model_loaded": True,
            "enter_seconds": _WHISPER_LOAD_S,
            "text": text,
            "transcript_entries": entries,
            "transcript_metadata": {
                "source": "whisper",
                "entries_count": len(entries),
                "s3_key": transcript_s3_key,
                "s3_url": transcript_s3_url,
                "whisper": whisper_info,
            },
            "processing_metrics": {
                "duration_seconds": round(time.perf_counter() - start_perf, 2),
                "gpu_type": "NVIDIA T4",
                "timings": timings,
            },
        }
    except Exception as e:
        print(f"❌ transcript_only ERROR: {e}")
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "task_id": task_id, "cold": cold,
        })
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# FRAME QUALITY + SELECTION HELPERS
# -----------------------------------------------------------------------------
def _save_jpeg(frame, path, long_side=1920):
    """Aspect-ratio-preserving resize to `long_side`, then JPEG q92.
    (The old code stretched everything to exactly 1920x1080, distorting any
    video that wasn't 16:9.)"""
    import cv2
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) > 0.01:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=interp)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _face_quality(frame, face):
    """
    Composite quality score for one detected face. The old code ranked frames
    by detector confidence alone, which saturates near 0.9 and knows nothing
    about blur, pose, or size — so "best frame" was often a blurry profile.

    Subscores (each normalized to 0..1):
      sharpness — Laplacian variance of the face crop (resized to 112px so the
                  number is comparable across face sizes)
      frontal   — from InsightFace 3D-landmark head pose (yaw/pitch)
      size      — face height relative to the frame
      exposure  — penalizes very dark / blown-out faces
    Faces cut off by the frame edge get a flat 25% penalty.
    """
    import cv2
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fw, x2), min(fh, y2)
    bw, bh = x2c - x1c, y2c - y1c
    if bw < 24 or bh < 24:
        return None  # too small to judge (or to use)

    crop = frame[y1c:y2c, x1c:x2c]
    gray = cv2.cvtColor(cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(1.0, (lap_var / 250.0) ** 0.5)

    mean_b = float(gray.mean())
    exposure = max(0.0, 1.0 - abs(mean_b - 120.0) / 120.0)

    pose = getattr(face, 'pose', None)
    if pose is not None:
        pitch, yaw = float(pose[0]), float(pose[1])
        frontal = max(0.0, 1.0 - abs(yaw) / 55.0) * max(0.0, 1.0 - abs(pitch) / 55.0)
    else:
        frontal = 0.5

    size = min(1.0, bh / (0.22 * fh))
    edge = 0.75 if (x1 <= 1 or y1 <= 1 or x2 >= fw - 2 or y2 >= fh - 2) else 1.0

    quality = float(face.det_score) * (
        0.35 * sharpness + 0.25 * frontal + 0.25 * size + 0.15 * exposure
    ) * edge

    return {
        "quality": quality,
        "sharpness": sharpness,
        "frontal": frontal,
        "size": size,
        "exposure": exposure,
    }


def _pick_top_frames(samples, k, min_gap):
    """Greedy top-k by quality with a minimum time gap between picks, so the
    3 images are distinct moments instead of 3 near-identical frames. Relaxes
    the gap if the person wasn't on screen long enough to satisfy it."""
    ranked = sorted(samples, key=lambda s: s['quality'], reverse=True)
    picked = []
    for s in ranked:
        if len(picked) >= k:
            break
        if all(abs(s['timestamp'] - p['timestamp']) >= min_gap for p in picked):
            picked.append(s)
    if len(picked) < k:
        for s in ranked:
            if len(picked) >= k:
                break
            if s not in picked:
                picked.append(s)
    return picked


# -----------------------------------------------------------------------------
# UNIFIED FUNCTION (GPU + WEB ENDPOINT)
# -----------------------------------------------------------------------------
@app.function(
    gpu="T4",
    # Video decode is CPU-bound; reserving cores keeps scan times consistent
    # instead of depending on how busy the host happens to be.
    cpu=4,
    timeout=3600,
    secrets=[modal.Secret.from_name("object-storage-credentials")],
    volumes={
        "/root/.insightface": model_cache,
        "/root/.cache/whisper": whisper_cache,
    },
    # 300s so a warm-up ping fired at "user clicked upload" keeps the container
    # hot through the upload + submit gap.
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST")
def process(request: dict) -> dict:
    """
    Video → top-3 quality-ranked images per unique person (+ representative
    frames, transcript via VTT or Whisper fallback, S3 archival, webhook).

    Extra request options (all optional):
      whisper:            "auto" (default: only when no VTT) | "always" | "never"
      whisper_model:      faster-whisper model name, default "large-v3-turbo"
      transcript_only:    true -> skip the face pipeline, return the transcript
                          as soon as Whisper finishes (fast path for the brief)
      language:           ISO code hint for Whisper (default: auto-detect)
      max_talent_images:  images per person, default 3 (1-5)
      representative_frames: true/false or a count — defaults to 10 for legacy
                          video_url/amazon_data requests, 0 (skipped, faster)
                          for bucket/location/filename requests

    R2-native input (all three instead of video_url):
      bucket:             bucket holding the source video
      location:           folder/prefix inside the bucket (e.g. "content/acc1/vid42")
      filename:           video object name (e.g. "video.mp4")
    In this mode the video is read via the S3 API (no public URL needed) and all
    outputs — frames + transcript.json, prefixed with the video's name — are
    written back to that same folder. Legacy video_url/amazon_data requests keep
    the old extracted-frames/{date}/{job_id}/ layout and filenames untouched.
    """
    import cv2
    import boto3
    import numpy as np
    import onnxruntime
    from sklearn.preprocessing import normalize
    import requests
    import webvtt

    start_perf = time.perf_counter()
    timings = {}

    # Warm/cold + container identity so the API can verify warmth in-band.
    # _COLD is True only for this container's first real request.
    task_id = os.environ.get("MODAL_TASK_ID", "")
    global _COLD
    cold = _COLD
    _COLD = False

    # 1. HARDWARE CHECK — fail loudly, never silently fall back to CPU
    if os.system("nvidia-smi > /dev/null 2>&1") != 0:
        print("❌ CRITICAL: No GPU attached!")
    if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        raise RuntimeError("❌ ONNX Runtime is missing CUDA. Aborting.")

    # 1b. WARM-UP PING — {"warm": true} boots the container and preloads both
    # models, then returns. No download, no S3 writes, no webhook. Fire it the
    # moment a user starts an upload; the container stays hot for
    # scaledown_window (5 min), so the real request runs at warm speed.
    if request.get("warm"):
        t0 = time.perf_counter()
        _get_face_app()
        if request.get("preload_whisper", True):
            # Preload the English model + the cheap language detector (the common
            # path). large-v3 loads lazily only when a non-English clip appears.
            _get_whisper(request.get("whisper_model", DEFAULT_WHISPER_MODEL))
            _get_whisper(request.get("lang_detect_model", LANG_DETECT_MODEL))
        return {
            "status": "warm",
            "task_id": task_id,
            "cold": cold,
            "model_loaded": len(_WHISPER_MODELS) > 0,
            "models_loaded": list(_WHISPER_MODELS.keys()),
            "face_loaded": _FACE_APP is not None,
            "enter_seconds": _WHISPER_LOAD_S,
            "ready_in_seconds": round(time.perf_counter() - t0, 2),
        }

    # 1c. TRANSCRIPT-ONLY FAST PATH — skip the whole face pipeline so the
    # response returns the moment Whisper finishes (unblocks the LLM brief).
    # Talent frames are produced by a separate full call to this endpoint.
    if request.get("transcript_only") or request.get("mode") == "transcript":
        return _transcribe_only(request, task_id=task_id, cold=cold)

    # 2. PARSE INPUTS (Amazon or Direct)
    metadata = request.get("metadata", {})
    video_url = request.get("video_url")
    transcript_url = request.get("transcript_url")

    whisper_mode = request.get("whisper", "auto")  # auto | always | never
    whisper_model_name = request.get("whisper_model", DEFAULT_WHISPER_MODEL)
    language = request.get("language")
    max_images = max(1, min(5, int(request.get("max_talent_images", 3))))

    # R2-native input: bucket + location + filename instead of a URL.
    bucket_override = request.get("bucket")
    location = (request.get("location") or "").strip().strip("/")
    source_filename = (request.get("filename") or "").strip().strip("/")
    storage_mode = bool(source_filename)

    # Representative frames cost ~2-4s (extra decodes + encodes + uploads), so
    # the R2-native realtime flow skips them by default; legacy requests keep
    # them for Make.com compatibility. Override with "representative_frames":
    # true/false or a count.
    rep_cfg = request.get("representative_frames")
    if rep_cfg is None:
        rep_count = 0 if storage_mode else 10
    elif rep_cfg is True:
        rep_count = 10
    else:
        rep_count = max(0, min(30, int(rep_cfg)))

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

    if not video_url and not storage_mode:
        return {"error": "No video_url found (send video_url, amazon_data, or bucket/location/filename)"}

    # 3. SETUP & DOWNLOAD
    # Object storage is env-driven (same names as the shared .env). With
    # OBJECT_STORAGE_ENDPOINT_URL set this targets Cloudflare R2 (S3-compatible
    # API); without it, it falls back to plain AWS S3 via boto3's default
    # credential chain.
    job_id = str(uuid.uuid4())
    date_folder = datetime.now().strftime("%Y-%m-%d")
    bucket_name = bucket_override or os.environ.get("OBJECT_STORAGE_BUCKET", "logie-users")
    endpoint_url = os.environ.get("OBJECT_STORAGE_ENDPOINT_URL")
    if endpoint_url:
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
    else:
        s3_client = boto3.client('s3')
    # Public URLs: R2 buckets aren't reachable at *.s3.amazonaws.com — they need
    # a public dev/custom domain (or the endpoint itself for private buckets).
    public_base = (
        os.environ.get("OBJECT_STORAGE_PUBLIC_BASE_URL")
        or f"https://{bucket_name}.s3.amazonaws.com"
    ).rstrip('/')

    # Output layout: storage mode drops results next to the source file with the
    # video's name as prefix (collision-safe if a folder holds several videos).
    # Legacy mode keeps the exact old layout so Make.com flows keep working.
    if storage_mode:
        # Tolerate a location that repeats the bucket name ("bucket/media/..."),
        # a natural mistake when copying paths from the R2 dashboard.
        if location == bucket_name or location.startswith(bucket_name + "/"):
            location = location[len(bucket_name):].strip("/")
        source_key = f"{location}/{source_filename}" if location else source_filename
        stem = os.path.splitext(source_filename)[0]
        base_path = location
        name_prefix = f"{stem}_"
    else:
        source_key = None
        base_path = f"extracted-frames/{date_folder}/{job_id}"
        name_prefix = ""

    def mk_key(fname):
        return f"{base_path}/{fname}" if base_path else fname

    def mk_url(key):
        # Percent-encode the key ("5 Star ... .mp4" has spaces) but keep slashes.
        from urllib.parse import quote
        return f"{public_base}/{quote(key)}"

    # Image uploads run on a small thread pool so R2 round-trips never block
    # the decode/inference loop. Markers get '_ok' set on success.
    from concurrent.futures import ThreadPoolExecutor
    uploader = ThreadPoolExecutor(max_workers=8)
    upload_futures = []

    def upload_async(local_p, key, marker=None, ctype='image/jpeg'):
        def _do():
            try:
                s3_client.upload_file(local_p, bucket_name, key, ExtraArgs={'ContentType': ctype})
                if marker is not None:
                    marker['_ok'] = True
            except Exception as e:
                print(f"⚠️ Upload failed for {key}: {e}")
        upload_futures.append(uploader.submit(_do))

    temp_dir = tempfile.mkdtemp()
    try:
        # --- A. Download Video ---
        t0 = time.perf_counter()
        video_path = os.path.join(temp_dir, "video.mp4")

        if storage_mode:
            print(f"⬇️ Downloading from bucket: {bucket_name}/{source_key}")
            try:
                s3_client.download_file(bucket_name, source_key, video_path)
            except Exception as e:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=404, content={
                    "status": "error",
                    "error": f"Could not fetch '{source_key}' from bucket '{bucket_name}': {e}",
                    "hint": "location must be the folder path inside the bucket (no bucket name), e.g. media/videos/toprated-videos",
                })
        elif "m3u8" in video_url or video_url.endswith(".m3u8"):
            print(f"⬇️ Downloading Video: {video_url}")
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
            print(f"⬇️ Downloading Video: {video_url}")
            r = requests.get(video_url, stream=True, timeout=60)
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        timings["download"] = round(time.perf_counter() - t0, 2)

        # --- B. Transcript: VTT first, Whisper as fallback ---
        t0 = time.perf_counter()
        transcript_data = []
        transcription_source = None
        whisper_info = None

        if transcript_url and whisper_mode != "always":
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
                if transcript_data:
                    transcription_source = "vtt"
            except Exception as e:
                print(f"⚠️ VTT transcript failed: {e}")
        timings["transcript_vtt"] = round(time.perf_counter() - t0, 2)

        # Whisper runs in a background thread OVERLAPPED with the face scan, so
        # transcription usually adds ~zero wall-clock time to the response.
        need_whisper = whisper_mode == "always" or (whisper_mode == "auto" and not transcript_data)
        whisper_thread = None
        whisper_out = {}
        if need_whisper:
            audio_path = os.path.join(temp_dir, "audio.wav")
            rc = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path],
                capture_output=True,
            ).returncode
            if rc == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
                import threading

                def _run_whisper():
                    t = time.perf_counter()
                    try:
                        model = _get_whisper(whisper_model_name)
                        print("🎙️ Transcribing with Whisper (parallel with scan)...")
                        segments, info = model.transcribe(
                            audio_path, beam_size=5, vad_filter=True, language=language,
                        )
                        entries = []
                        for seg in segments:
                            txt = seg.text.strip()
                            if txt:
                                entries.append({
                                    'start': round(seg.start, 2),
                                    'end': round(seg.end, 2),
                                    'speaker': None,
                                    'text': txt,
                                })
                        whisper_out['entries'] = entries
                        whisper_out['info'] = info
                    except Exception as e:
                        whisper_out['error'] = str(e)
                    whisper_out['seconds'] = round(time.perf_counter() - t, 2)

                whisper_thread = threading.Thread(target=_run_whisper, daemon=True)
                whisper_thread.start()
            else:
                print("⚠️ No usable audio track — skipping Whisper.")

        # --- C. Video Metadata ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not fps or fps <= 0 or fps != fps:  # 0/NaN guard
            fps = 30.0
        duration = total_frames / fps if total_frames > 0 else 0

        print(f"📊 Stats: {duration:.1f}s | {fps:.1f} FPS | {v_w}x{v_h}")

        # --- D. Initialize GPU inference (cached across warm requests) ---
        face_app = _get_face_app()

        # Sampling density: short videos get sampled harder — more candidates
        # per person means better clustering AND better best-frame choices.
        if duration <= 0:
            interval = 1.0
        elif duration <= 90:
            interval = 1.0 / 3.0
        elif duration <= 600:
            interval = 1.0
        else:
            interval = 2.0
        sample_stride = max(1, int(round(fps * interval)))

        # Representative frame indices (start / spread / end), as before.
        if rep_count <= 0:
            sorted_rep = []
        elif total_frames > 1:
            rep_indices = set([0, min(29, total_frames - 1), max(0, total_frames - 31), max(0, total_frames - 2)])
            if total_frames > 10:
                for p in np.linspace(30, total_frames - 31, 10).astype(int)[1:-1]:
                    rep_indices.add(int(max(0, p)))
            sorted_rep = sorted(rep_indices)[:rep_count]
        else:
            sorted_rep = [i * sample_stride for i in range(rep_count)]
        rep_set = set(sorted_rep)

        # --- E+F. SINGLE SEQUENTIAL PASS: rep frames + face scan ---
        # cap.grab() skips decode-to-image for frames we don't need; no random
        # seeks. The old per-sample cap.set() forced a keyframe seek + decode
        # for every sample — the main reason runs were slow despite the GPU.
        t0 = time.perf_counter()
        rep_results = []
        all_faces = []
        sampled_frames = 0

        cap = cv2.VideoCapture(video_path)
        f_idx = 0
        while True:
            if not cap.grab():
                break
            is_rep = f_idx in rep_set
            is_sample = (f_idx % sample_stride == 0)
            if is_rep or is_sample:
                ret, frame = cap.retrieve()
                if ret:
                    if is_rep:
                        i = sorted_rep.index(f_idx)
                        fname = f"{name_prefix}representative_frame_{i}.jpg"
                        key = mk_key(fname)
                        local_p = os.path.join(temp_dir, fname)
                        _save_jpeg(frame, local_p)
                        rep_entry = {
                            "frame_index": f_idx,
                            "filename": fname,
                            "s3_key": key,
                            "s3_url": mk_url(key),
                            "timestamp": round(f_idx / fps, 2),
                            "_ok": False,
                        }
                        rep_results.append(rep_entry)
                        upload_async(local_p, key, marker=rep_entry)
                    if is_sample:
                        sampled_frames += 1
                        for face in face_app.get(frame):
                            if face.det_score <= 0.6:
                                continue
                            q = _face_quality(frame, face)
                            if q is None:
                                continue
                            all_faces.append({
                                'embedding': face.embedding,
                                'score': float(face.det_score),
                                'frame_idx': f_idx,
                                'timestamp': f_idx / fps,
                                **q,
                            })
            f_idx += 1
        cap.release()
        if duration <= 0:
            duration = f_idx / fps
        timings["scan"] = round(time.perf_counter() - t0, 2)
        print(f"🔍 Scanned {sampled_frames} frames, {len(all_faces)} face samples in {timings['scan']}s")

        # Collect the parallel Whisper result (usually already finished by now).
        if whisper_thread is not None:
            whisper_thread.join()
            timings["whisper"] = whisper_out.get('seconds')
            if whisper_out.get('error'):
                print(f"⚠️ Whisper failed: {whisper_out['error']}")
            elif whisper_out.get('entries'):
                transcript_data = whisper_out['entries']
                transcription_source = "whisper"
                info = whisper_out['info']
                whisper_info = {
                    "model": whisper_model_name,
                    "language": info.language,
                    "language_probability": round(float(info.language_probability), 3),
                }
                print(f"🎙️ Whisper: {len(transcript_data)} segments, lang={info.language}")

        # --- G. Cluster identities & pick top-N frames per person ---
        t0 = time.perf_counter()
        talent_results = []

        def get_context(ts):
            for e in transcript_data:
                if e['start'] <= ts <= e['end']:
                    return e['speaker'], e['text']
            for e in transcript_data:
                if abs(e['start'] - ts) < 2.0:
                    return e['speaker'], e['text']
            return None, None

        if all_faces:
            print(f"🧠 Clustering {len(all_faces)} faces...")
            feats = normalize(np.array([f['embedding'] for f in all_faces]))
            labels = cluster_identities(feats)

            groups = {}
            for i, l in enumerate(labels):
                if l != -1:
                    groups.setdefault(int(l), []).append(all_faces[i])

            # Primary talent first: most screen time, then biggest average face.
            order = sorted(
                groups,
                key=lambda pid: (-len(groups[pid]), -float(np.mean([s['size'] for s in groups[pid]]))),
            )

            min_gap = max(1.0, min(5.0, duration * 0.05))
            save_jobs = {}  # frame_idx -> [{local, img}]

            for rank, pid in enumerate(order):
                samples = groups[pid]
                picks = _pick_top_frames(samples, max_images, min_gap)
                images = []
                for j, s in enumerate(picks):
                    fname = f"{name_prefix}person_{rank}.jpg" if j == 0 else f"{name_prefix}person_{rank}_alt{j}.jpg"
                    key = mk_key(fname)
                    img = {
                        "filename": fname,
                        "s3_key": key,
                        "s3_url": mk_url(key),
                        "timestamp": round(s['timestamp'], 2),
                        "frame_index": s['frame_idx'],
                        "quality": round(s['quality'], 3),
                        "detection_score": round(s['score'], 3),
                        "sharpness": round(s['sharpness'], 3),
                        "frontal": round(s['frontal'], 3),
                        "_ok": False,
                    }
                    images.append(img)
                    save_jobs.setdefault(s['frame_idx'], []).append(
                        {"local": os.path.join(temp_dir, fname), "img": img}
                    )

                best = picks[0]
                speaker, context = get_context(best['timestamp'])
                stamps = [s['timestamp'] for s in samples]
                talent_results.append({
                    "person_id": rank,
                    "is_primary": rank == 0,
                    "name": speaker if speaker else f"Person {rank}",
                    "context_text": context,
                    "timestamp": round(best['timestamp'], 2),
                    "score": round(best['score'], 2),
                    "quality": round(best['quality'], 3),
                    "appearances": len(samples),
                    "screen_time_seconds": round(len(samples) * interval, 1),
                    "first_seen": round(min(stamps), 2),
                    "last_seen": round(max(stamps), 2),
                    "images": images,
                })

            # Re-fetch only the chosen frames (a handful of targeted seeks).
            cap = cv2.VideoCapture(video_path)
            for fi in sorted(save_jobs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue
                for job in save_jobs[fi]:
                    _save_jpeg(frame, job["local"])
                    upload_async(job["local"], job["img"]["s3_key"], marker=job["img"])
            cap.release()

        # Wait for every queued upload (rep + talent) to land, then drop
        # anything that failed to upload or re-read.
        for f in upload_futures:
            f.result()
        rep_results = [
            {k: v for k, v in r.items() if k != "_ok"}
            for r in rep_results if r.get("_ok")
        ]
        for t in talent_results:
            t["images"] = [
                {k: v for k, v in img.items() if k != "_ok"}
                for img in t["images"] if img["_ok"]
            ]
        talent_results = [t for t in talent_results if t["images"]]
        for t in talent_results:
            t["filename"] = t["images"][0]["filename"]
            t["s3_key"] = t["images"][0]["s3_key"]
            t["s3_url"] = t["images"][0]["s3_url"]
        timings["cluster_select"] = round(time.perf_counter() - t0, 2)

        # --- H. Archive Full Video (skipped in storage mode — it's already there) ---
        if storage_mode:
            vid_key = source_key
            video_s3_url = mk_url(source_key)
        else:
            content_id = metadata.get('aci_content_id') or metadata.get('content_id') or job_id
            vid_filename = f"{content_id}.mp4"
            vid_key = f"amazon-shorts/{date_folder}/{vid_filename}"

            print(f"💾 Archiving video to {vid_key}")
            try:
                s3_client.upload_file(video_path, bucket_name, vid_key, ExtraArgs={
                    'ContentType': 'video/mp4',
                    'Metadata': {'job_id': job_id}
                })
                video_s3_url = mk_url(vid_key)
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

        # Full transcript to S3 (payload keeps only the truncated preview).
        transcript_s3_key = None
        transcript_s3_url = None
        if transcript_data:
            try:
                transcript_s3_key = mk_key(f"{name_prefix}transcript.json")
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=transcript_s3_key,
                    Body=json.dumps({
                        "source": transcription_source,
                        "whisper": whisper_info,
                        "entries": transcript_data,
                    }, ensure_ascii=False).encode("utf-8"),
                    ContentType='application/json',
                )
                transcript_s3_url = mk_url(transcript_s3_key)
            except Exception as e:
                print(f"⚠️ Transcript upload failed: {e}")

        # --- J. Construct Payload (original fields preserved, new ones added) ---
        result = {
            "status": "success",
            "job_id": job_id,
            "task_id": task_id,
            "cold": cold,
            "date_folder": date_folder,
            "base_path": base_path,
            "metadata": metadata or {},
            "processing_metrics": {
                "duration_seconds": proc_time,
                "estimated_cost_usd": gpu_cost,
                "gpu_type": "NVIDIA T4",
                "timings": timings,
                "sampled_frames": sampled_frames,
                "face_samples": len(all_faces),
                "sample_interval_seconds": round(interval, 2),
            },
            "video_metadata": {
                "duration_seconds": round(duration, 2),
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "resolution": f"{v_w}x{v_h}",
                "source_url": video_url or mk_url(source_key),
                "archived_s3_key": vid_key,
                "archived_s3_url": video_s3_url
            },
            "transcript_metadata": {
                "source": transcription_source,
                "source_url": transcript_url,
                "entries_count": len(transcript_data),
                "s3_key": transcript_s3_key,
                "s3_url": transcript_s3_url,
                "whisper": whisper_info,
            } if transcript_data else None,
            "transcription": transcription_text,
            "transcript_entries": transcript_data[:1000],
            "talent_count": len(talent_results),
            "talent_frames": talent_results,
            "best_talent_images": talent_results[0]["images"] if talent_results else [],
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
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "job_id": job_id,
        })

    finally:
        uploader.shutdown(wait=False, cancel_futures=True)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
