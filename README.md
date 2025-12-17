# Video Talent Frame Extractor

Extract the "best" frame of every unique person in a video using **InsightFace** AI and **DBSCAN** deduplication.

## 🚀 Deployment Options

| Option | Speed | Best For |
|--------|-------|----------|
| **Modal (GPU)** ⭐ | ~10-20x faster | Production, Long Videos |
| **Lambda (CPU)** | Slower | Low volume, Short Videos |

### Recommended: Modal GPU

For production use, we strongly recommend **Modal.com** for GPU-accelerated processing.

👉 **See [MODAL_BLUEPRINT.md](./MODAL_BLUEPRINT.md) for the complete setup guide.**

Quick start:
```bash
pip install modal
modal setup
modal secret create aws-s3-credentials AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
modal deploy modal_app.py
```

---

## 🎯 Features

-   **Tiered Processing**: Fast linear scan for short videos, smart 2-pass for long videos.
-   **Smart Deduplication**: Clusters faces to identify unique people.
-   **Quality Scoring**: Picks the best frame per person (pose, size, confidence).
-   **Standardized Output**: 1080p resolution (1920x1080 or 1080x1920).

## 🛠️ Architecture

```
Input:  {"bucket": "my-bucket", "key": "video.mp4"}
Output: processed/{video_name}/person_{id}.jpg → Your S3 Bucket
```

---

## 📦 Lambda Deployment (Alternative)

If you prefer AWS Lambda (CPU-only):

### Prerequisites
- Docker
- AWS CLI configured

### Deploy

```bash
# Build & push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker build -t video-talent-extractor .
docker tag video-talent-extractor:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/video-talent-extractor:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/video-talent-extractor:latest
```

### Lambda Config
- **Memory**: 4-8GB
- **Timeout**: 5-10 min
- **Permissions**: `s3:GetObject`, `s3:PutObject`

---

## 🧪 Local Testing

```bash
# Install deps (needs C++ build tools on Windows)
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Create run_local.py (not in repo)
# Then run:
python run_local.py
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [`MODAL_BLUEPRINT.md`](./MODAL_BLUEPRINT.md) | Complete Modal GPU setup guide |
| `modal_app.py` | Modal implementation (GPU) |
| `app.py` | Lambda implementation (CPU) |
| `Dockerfile` | Lambda container config |

---

## 💰 Cost Comparison

| Platform | 10-min Video |
|----------|--------------|
| Modal (T4 GPU) | ~$0.01 |
| Lambda (10GB RAM) | ~$0.05 |
