FROM public.ecr.aws/lambda/python:3.12

# Install FFmpeg
RUN dnf install -y ffmpeg libSM libXext && dnf clean all

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace models to /home/sbx_user1051/.insightface (default)
# To avoid permissions issues in Lambda, we need to ensure these are accessible.
# However, Lambda runs as a non-root user often, and only /tmp is writable.
# Best practice: Download models during build and set INSIGHTFACE_HOME to a location we control
# OR set HOME to /tmp at runtime and download there (slower).
# Let's try to pre-download.
ENV INSIGHTFACE_HOME=${LAMBDA_TASK_ROOT}/.insightface
RUN python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l', root='${INSIGHTFACE_HOME}'); app.prepare(ctx_id=0, det_size=(640, 640))"

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]
