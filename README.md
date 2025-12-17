# Video Talent Frame Extractor (Lambda)

A high-performance, serverless AWS Lambda function designed to extract the "best" frame of every unique person in a video. Built with **InsightFace** for detection and **DBSCAN** for deduplication.

## 🚀 Features

-   **Tiered Processing Logic**:
    -   **Short Videos (< 5 min)**: Blazing fast linear scan (1 check/sec).
    -   **Long Videos (> 5 min)**: Smart 2-Pass Optimization (Coarse Scan + Dense Refinement) to minimize compute costs on long content.
-   **Smart Deduplication**: Uses DBSCAN clustering on facial embeddings to group unique people and ignore duplicates.
-   **Quality Scoring**: Selects the best frame based on detection confidence, pose (looking at camera), and size.
-   **Standardized Output**: Automatically corrects aspect ratios to standard 1080p (1920x1080 or 1080x1920).
-   **Serverless**: Runs on AWS Lambda (Container Image) for infinite scalability.

## 🛠️ Architecture

1.  **Input**: JSON Payload `{"bucket": "my-bucket", "key": "video.mp4"}`.
2.  **Process**:
    -   Downloads video to `/tmp`.
    -   analyzes frames using `insightface`.
    -   Clusters faces to find unique identities.
    -   Resizes and uploads the best shot for each person to S3.
3.  **Output**: `processed/{original_filename}/person_{id}.jpg` in the source bucket.

## 📦 Deployment Guide

### Prerequisites
-   Docker installed.
-   AWS CLI configured with permissions for ECR and Lambda.

### 1. Build & Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -t video-talent-extractor .

# Tag & Push
docker tag video-talent-extractor:latest <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/video-talent-extractor:latest
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/video-talent-extractor:latest
```

### 2. Create Lambda Function

-   **Type**: Container Image.
-   **Memory**: 4GB - 8GB (Recommended for speed).
-   **Timeout**: 5-10 mins (depends on max video length).
-   **Ephemeral Storage**: > 512MB (Recommended 2GB+ for large video downloads).
-   **Permissions**:
    -   `s3:GetObject` (Source bucket)
    -   `s3:PutObject` (Source bucket)

## 🧪 Local Testing

To test the logic locally without deploying, you can use a python script that mocks the Lambda context.

**1. Install Dependencies locally**
You will need C++ Build Tools (VS Code users on Windows: install "Desktop development with C++" workload).
```bash
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Create a Run Script (`run_local.py`)**
*Note: This file is excluded from git to keep the repo clean.*

```python
import app
import logging

# ... Mock Context ...
event = {
    "bucket": "test-bucket",
    "key": "test-video.mp4"
}
app.lambda_handler(event, None)
```

**3. Run**
```bash
python run_local.py
```

## 💡 Best Practices & Configuration

### Memory vs Speed
InsightFace is CPU intensive.
-   **2GB Memory**: Functional but slow (~2-3fps processing).
-   **10GB Memory**: Much faster, as Lambda allocates proportional vCPU threads (up to 6 vCPUs).

### Tiered Logic Customization
The 5-minute threshold is hardcoded in `app.py`.
-   To adjust: Search for `if duration < 300:` and modify the usage.

### Model Storage
The Dockerfile does NOT bake the models in to keep the image small (~?GB). Instead, `insightface` downloads them on first run to `/tmp`.
-   **Optimization**: Mount an **EFS** volume to `/tmp/.insightface` to persist models across cold starts, saving ~5 seconds per cold start.
