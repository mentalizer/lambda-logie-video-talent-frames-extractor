# Modal GPU Blueprint - Complete Beginner's Guide

This guide walks you through deploying the Video Talent Frame Extractor on **Modal.com** with GPU acceleration.

---

## 🎯 What is Modal?

Modal is a **serverless GPU cloud**. Think of it like AWS Lambda, but with GPUs and zero DevOps.

- **No Docker registries** to manage
- **No Kubernetes** or EC2 instances
- **Pay-per-second** (~$0.0016/sec for T4 GPU)
- **Auto-scaling** from 0 to 100+ instances

---

## 📦 Storage: Where Do Frames Go?

**Your frames stay in YOUR S3 bucket.** Modal does NOT store your data.

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
│  Your App   │────────▶│  Modal Function  │────────▶│  Your S3    │
│  (API Call) │         │  (GPU Instance)  │         │  Bucket     │
└─────────────┘         └──────────────────┘         └─────────────┘
                              │                             ▲
                              │  1. Downloads video         │
                              │  2. Processes frames        │
                              │  3. Uploads results ────────┘
```

---

## 🔐 Credential Security

**Your AWS credentials are stored securely as a Modal Secret.**

Modal Secrets are:
- Encrypted at rest
- Never exposed in logs
- Injected as environment variables only during function execution
- Not accessible to Modal staff

### Setting Up Your Secret

```bash
# After installing Modal CLI:
modal secret create aws-s3-credentials \
  AWS_ACCESS_KEY_ID=AKIA... \
  AWS_SECRET_ACCESS_KEY=wJal... \
  AWS_REGION=us-east-1
```

> **Best Practice**: Create a dedicated IAM user with ONLY `s3:GetObject` and `s3:PutObject` permissions for your specific bucket. Never use root credentials.

---

## 🚀 Step-by-Step Setup

### 1. Install Modal CLI

```bash
pip install modal
modal setup  # Opens browser to authenticate
```

### 2. Create AWS Secret

Create an IAM user with this policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
    }
  ]
}
```

Then store the credentials in Modal:
```bash
modal secret create aws-s3-credentials \
  AWS_ACCESS_KEY_ID=<your-key> \
  AWS_SECRET_ACCESS_KEY=<your-secret> \
  AWS_REGION=us-east-1
```

### 3. Create Model Cache Volume

```bash
modal volume create insightface-models
```

This stores InsightFace models so they don't download every time.

### 4. Deploy the App

```bash
modal deploy modal_app.py
```

You'll see output like:
```
✓ Created web endpoint: https://your-user--video-talent-extractor-process-video.modal.run
```

---

## 🧪 Testing Your Endpoint

### Via cURL

```bash
curl -X POST https://your-user--video-talent-extractor-process-video.modal.run \
  -H "Content-Type: application/json" \
  -d '{"bucket": "my-bucket", "key": "videos/test.mp4"}'
```

### Via Python

```python
import requests

response = requests.post(
    "https://your-user--video-talent-extractor-process-video.modal.run",
    json={"bucket": "my-bucket", "key": "videos/test.mp4"}
)
print(response.json())
```

### Response Example

```json
{
  "status": "success",
  "people_found": 3,
  "results": [
    {
      "person_id": 0,
      "s3_key": "processed/test/person_0.jpg",
      "s3_url": "https://my-bucket.s3.amazonaws.com/processed/test/person_0.jpg",
      "timestamp": 45.2,
      "score": 78.5
    },
    ...
  ]
}
```

---

## 💰 Cost Estimation

| Video Length | GPU Time | Cost (~) |
|--------------|----------|----------|
| 1 minute     | ~5 sec   | $0.002   |
| 10 minutes   | ~30 sec  | $0.012   |
| 90 minutes   | ~2 min   | $0.19    |

Modal charges $0.000376/sec for T4 GPU.

---

## 🔧 Customization

### Change GPU Type

```python
@app.function(gpu="A10G")  # Faster, ~$0.001/sec
```

Options: `T4` (cheapest), `L4`, `A10G`, `A100` (fastest)

### Add Authentication

For production, add an API key check:

```python
@modal.web_endpoint(method="POST")
def process_video(request: dict) -> dict:
    if request.get("api_key") != os.environ.get("MY_API_KEY"):
        return {"error": "Unauthorized"}, 401
    # ... rest of code
```

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `modal_app.py` | Main Modal application with GPU function and endpoint |

---

## 🤔 FAQ

**Q: Is my data safe?**
A: Yes. Modal doesn't store your data. Videos are downloaded to ephemeral storage, processed, then deleted. Results go to YOUR S3.

**Q: What if I want results in Modal's cloud instead of S3?**
A: You could use Modal Volumes, but S3 is better for integration with your existing systems. Modal Volumes are meant for caching, not permanent storage.

**Q: Can I run this locally first?**
A: Yes! Use `modal run modal_app.py --bucket x --key y` for a one-off test run.

---

## 🎉 You're Ready!

1. ✅ Modal CLI installed
2. ✅ AWS Secret configured
3. ✅ Model cache volume created
4. ✅ App deployed

Your endpoint is live. Call it from your Logie backend whenever you need to process a video!
