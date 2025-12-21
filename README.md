# 🎥 Video Talent Frame Extractor (Modal GPU)

High-performance, serverless video processing for extracting the **best** frames of people and representative shots. Powered by **NVIDIA T4 GPUs**, **InsightFace AI**, and **Modal.com**.

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

## 📡 API & Webhook

### Input (POST)
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
-   [`modal_app.py`](./modal_app.py): The core GPU logic & engine.
-   [`MODAL_BLUEPRINT.md`](./MODAL_BLUEPRINT.md): Beginner-friendly setup guide.
-   [`app.py`](./app.py): Legacy Lambda version (CPU-only).
