# 🎥 Video Talent Frame Extractor (Modal GPU)


#######

FINAL MAIN FILE (deploy this one):
main.py  →  `modal deploy main.py`

#######

## ✨ What main.py does now

- **Top-3 talent images per person** — every unique person gets up to 3 quality-ranked
  images (`person_0.jpg`, `person_0_alt1.jpg`, `person_0_alt2.jpg`), each from a
  distinct moment in the video. The response also includes `best_talent_images`
  (the 3 best images of the primary talent) for easy consumption.
- **Real quality scoring** — frames are ranked by a composite of face sharpness
  (Laplacian), head pose frontal-ness (3D landmarks), face size, exposure, and
  detector confidence — not detector confidence alone.
- **Primary talent detection** — `person_0` is always the person with the most
  screen time (`is_primary: true`, plus `appearances` / `screen_time_seconds`).
- **Fast single-pass scan** — one sequential decode pass using `grab()`/`retrieve()`
  instead of thousands of random seeks; representative frames are captured in the
  same pass. Models load once per container and are reused across warm requests.
- **Whisper transcription (GPU, parallel)** — if a video has no VTT captions, audio
  is transcribed with faster-whisper `large-v3` on the same T4, **in a background
  thread overlapped with the face scan** so it adds ~zero wall-clock time. Control
  it per request: `"whisper": "auto" | "always" | "never"`,
  `"whisper_model": "large-v3"`, `"language": "en"`. Full transcript is uploaded
  as `transcript.json`.
- **No more stretched images** — output frames keep their aspect ratio
  (long side 1920) instead of being forced to 1920×1080.

### New request options (all optional)

```json
{
  "video_url": "https://.../video.mp4",
  "whisper": "auto",
  "whisper_model": "large-v3",
  "language": "en",
  "max_talent_images": 3
}
```

## ☁️ Cloudflare R2 (replaces hardcoded S3)

`main.py` reads object storage from the `object-storage-credentials` Modal secret:

```bash
modal secret create object-storage-credentials \
  OBJECT_STORAGE_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com \
  OBJECT_STORAGE_PUBLIC_BASE_URL=https://<pub-xxxx.r2.dev or custom domain> \
  OBJECT_STORAGE_BUCKET=<default-bucket> \
  OBJECT_STORAGE_ACCESS_KEY_ID=<R2 API token key id> \
  OBJECT_STORAGE_SECRET_ACCESS_KEY=<R2 API token secret>
```

If `OBJECT_STORAGE_ENDPOINT_URL` is unset the code falls back to plain AWS S3
(old behavior). All returned URLs are built from `OBJECT_STORAGE_PUBLIC_BASE_URL`.

### R2-native requests: process a file already in the bucket

Send `bucket` + `location` + `filename` instead of `video_url` — the video is read
via the S3 API (no public URL needed) and all outputs land **in the same folder**,
prefixed with the video's name:

```json
{
  "bucket": "logie-users",
  "location": "content/acc-123/vid-42",
  "filename": "video.mp4"
}
```

Produces `content/acc-123/vid-42/video_person_0.jpg`, `video_person_0_alt1.jpg`,
`video_transcript.json`, … next to the source video. Representative frames are
**skipped by default in this mode** for speed — request them with
`"representative_frames": 10`. Legacy `video_url`/`amazon_data` requests keep
the old `extracted-frames/{date}/{job_id}/` layout, filenames, and 10
representative frames, so existing Make.com flows are unaffected.


High-performance, serverless video processing for extracting the **best** frames of people and representative shots. Powered by **NVIDIA T4 GPUs**, **InsightFace AI**, and **Modal.com**.

## 📋 **Available Versions:**

- **`modal_app.py`** - Full-featured version with account/content folder organization
- **`video_only_extractor.py`** - **NEW!** Simplified job-based version with date/UUID organization

---

## 🚀 Key Achievements & Features

-   **⚡ 20x Speed Increase**: Migrated from Lambda CPU to **Modal GPU (NVIDIA T4)**. A 3-minute video processes in seconds, not minutes.
-   **📡 Optimized Downloads**: Downloads videos from S3 for reliable processing with OpenCV. Fast and reliable metadata extraction.
-   **🎯 Single-Pass "Seek Scan"**: Optimized algorithm that "jumps" through frames (1 FPS) instead of linear reading. Drastically reduces network latency.
-   **🖼️ Exact 10 Representative Frames**: Robust logic guarantees exactly 10 high-quality representative frames for any video length.
-   **💰 Cost & Performance Tracking**: Webhook payload now includes **processing time** and **estimated GPU cost** (typically <$0.01 per run).
-   **🤝 Smarter Clustered Detection**: Uses **DBSCAN** to group faces and pick the single best quality frame per unique person found.
-   **📁 Organized Storage**: Frames are now saved in dedicated folders using the structure `{bucket}/content/{account_id}/{content_id}/extraction-talent-frames/` to prevent overwriting between videos.
-   **🧠 Memory Optimized**: Processes videos with minimal memory footprint, preventing resource limits on long videos.

---

## 🛠️ Quick Setup (Modal GPU)

The recommended way to run this is using **Modal.com**.

1.  **Install & Auth**:
    ```bash
    pip install modal
    modal setup
    ```
2.  **Configure S3 Credentials**:
    ```bash
    modal secret create aws-s3-credentials AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
    ```
3.  **Create Cache Volume** (One-time):
    ```bash
    modal volume create insightface-models
    ```
4.  **Deploy**:
    ```bash
    modal deploy modal_app.py
    ```

👉 **See [MODAL_BLUEPRINT.md](./MODAL_BLUEPRINT.md) for the complete setup guide.**

---

## 🎯 **Video-Only Extractor (NEW!)**

Simplified job-based version perfect for straightforward video processing.

### **Features:**
- ✅ Direct video/transcript URLs (no S3 keys required)
- ✅ Job-based organization: `extracted-frames/{date}/{job_uuid}/`
- ✅ Easy searching by date or job ID
- ✅ Memory optimized for reliability
- ✅ Same AI processing power
- ✅ **Full video archiving** to `logie-users/amazon-shorts/` folder

### **Quick Deploy:**
```bash
modal deploy video_only_extractor.py
```

### **API Usage:**

#### **Option 1: Direct URLs**
```bash
curl --location 'https://mentalizer--video-only-extractor-process-video-job.modal.run' \
--header 'Content-Type: application/json' \
--data '{
  "video_url": "https://example.com/video.mp4",
  "transcript_url": "https://example.com/transcript.vtt",
  "metadata": {
    "job_name": "My Webinar",
    "user_id": "12345"
  }
}'
```

#### **Option 2: Amazon Live Data** ⭐ **Recommended for your use case**

### **GPU Acceleration** 🚀

This version is configured to run on **NVIDIA T4 GPUs** for maximum speed. Face detection should process videos in **seconds rather than minutes**. If you see CPU fallback warnings in the logs, the GPU libraries may need adjustment.

#### **HLS Testing Endpoint:**
```bash
curl --location 'https://mentalizer--video-only-extractor-test-hls-url.modal.run' \
--header 'Content-Type: application/json' \
--data '{"hls_url": "YOUR_HLS_URL_HERE"}'
```

**Returns:** HLS playlist info and segment accessibility status.
```bash
curl --location 'https://mentalizer--video-only-extractor-process-video-job.modal.run' \
--header 'Content-Type: application/json' \
--data '{
  "amazon_data": {
    "broadcast_id": "02c4ee7e633246e384019e387bbf6db4",
    "shop_id": "influencer-7feb78c5",
    "broadcast_title": "Watch this preview to check it out!",
    "hls_url": "https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/97dfe7ee-b7e4-4c31-ad90-6cdae1ea6cf9/default.jobtemplate.hls.m3u8",
    "closed_captions": "en,https://m.media-amazon.com/images/S/vse-vms-closed-captions-artifact-us-east-1-prod/closedCaptions/607dcc73-a36d-4e17-be03-a3c9de47142a.vtt",
    "video_preview_assets": [
      {
        "url": "https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/c8f1be9a-b050-4ed4-90b7-cf6f47cc5be0/videopreview.jobtemplate.mp4.default.mp4",
        "type": "default",
        "mimeType": "video/mp4"
      }
    ],
    "aci_content_id": "amzn1.vse.video.02c4ee7e633246e384019e387bbf6db4",
    "formatted_duration": "0:58"
  },
  "metadata": {
    "custom_field": "additional_metadata"
  }
}'
```

**Auto-extraction:** The system automatically extracts:
- **Video URL Priority** (optimized for short 10-30 second videos):
  1. **HLS Stream** from `hls_url` (preferred for short videos - auto-converted to MP4)
  2. **MP4 Preview** from `video_preview_assets`
  3. **Full MP4** constructed from `broadcast_id`
- **Transcript URL**: Parses VTT URL from `closed_captions` field
- **Metadata**: Merges Amazon data with your custom metadata

**HLS Processing:** Short HLS streams are automatically downloaded and converted to MP4 for OpenCV compatibility.

**Video Archiving:** Full processed videos are archived to `logie-users/amazon-shorts/{date}/` for future reuse.

**Troubleshooting HLS:**
- Use the test endpoint above to verify HLS URL accessibility
- Check logs for detailed error messages and fallback attempts
- HLS downloads include automatic fallback to direct URL access

**URL Verification:** Test constructed URLs with:
```bash
curl -I "https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/02c4ee7e633246e384019e387bbf6db4/default.jobtemplate.mp4"
```

### **Output Structure:**
```
logie-users/extracted-frames/
├── 2024-12-20/
│   ├── abc123-def456-789/
│   │   ├── person_0.jpg
│   │   ├── person_1.jpg
│   │   └── representative_frame_0.jpg
│   └── def789-ghi012-345/
│       └── person_0.jpg
└── 2024-12-21/
    └── ...
```

**Perfect for:** Simple video processing without complex folder hierarchies.

---

## 📡 API & Webhook

### Endpoint
**POST** `https://mentalizer--video-talent-extractor-process-video.modal.run`

### Authentication
Currently, the endpoint is **public** for ease of integration with your webhooks. If you require a `Bearer` token or `X-API-Key` header for production security, we can easily add a validation decorator.
```json
{
  "bucket": "logie-users",
  "main_folder": "content",
  "account_id": "4b6ccb29-e5bb-46fa-a516-19eca622c258",
  "content_id": "webinars/12-20-2025/81080519584-audio_transcript",
  "video_key": "content/4b6ccb29-e5bb-46fa-a516-19eca622c258/webinars/12-20-2025/81080519584-video.mp4",
  "transcript_key": "content/4b6ccb29-e5bb-46fa-a516-19eca622c258/webinars/12-20-2025/81080519584-audio_transcript.VTT",   // Optional
  "custom_metadata": {                                              // Optional
    "job_id": "123",
    "priority": "high",
    "youtube_video_id": "abc123"
  }
}
```

### Output (Webhook)
The results include:
-   **`custom_metadata`**: Exact copy of what you sent in the request.
-   **`talent_frames`**:
    -   `name`: Speaker name from VTT (or "Person X").
    -   `context_text`: The exact sentence spoken at the frame's timestamp.
-   **Processing Metrics**: Duration in seconds and estimated USD cost.

---

## 🧪 Local Testing
Run a quick test from your machine targeting a remote S3 file:
```bash
modal run modal_app.py --bucket "my-bucket" --main_folder "content" --account_id "account-uuid" --content_id "content-uuid" --video_key "path/to/video.mp4"
```
E.g. .\.venv\Scripts\modal run modal_app.py --bucket "logie-users" --main_folder "content" --account_id "4b6ccb29-e5bb-46fa-a516-19eca622c258" --content_id "webinars/12-20-2025/81080519584-audio_transcript" --video_key "content/4b6ccb29-e5bb-46fa-a516-19eca622c258/webinars/12-20-2025/81080519584-video.mp4"

---

## 💰 Performance vs Cost

| Video Length | CPU (Lambda) | GPU (Modal T4) | Est. Cost | Max Timeout | Memory Usage |
|--------------|--------------|----------------|-----------|------------|-------------|
| 3 Minutes    | ~2-3 Minutes | **~15 Seconds** | **$0.006** | 1 Hour     | Low         |
| 30 Minutes   | Timeout      | **~2 Minutes**  | **$0.040** | 1 Hour     | Medium      |
| 90 Minutes   | N/A          | **~15 Minutes** | **$0.200** | 1 Hour     | Optimized   |

---

## ⚠️ **Modal Limits & Timeouts:**

- **Function Timeout:** 1 hour maximum per function call
- **Free Tier:** Limited GPU hours per month
- **Paid Tier:** Higher limits available
- **For very long videos:** Consider splitting into segments or upgrading your Modal plan

---

## 📚 Repository Map
-   [`modal_app.py`](./modal_app.py): Full-featured version with account/content organization.
-   [`video_only_extractor.py`](./video_only_extractor.py): **NEW!** Simplified job-based version.
-   [`MODAL_BLUEPRINT.md`](./MODAL_BLUEPRINT.md): Beginner-friendly setup guide.
-   [`app.py`](./app.py): Legacy Lambda version (CPU-only).
