FROM python:3.10-slim

# ── Build argument: MLflow Run ID injected at build time ──────────────────────
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

# ── Install runtime dependencies ─────────────────────────────────────────────
RUN pip install --no-cache-dir mlflow scikit-learn

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Simulate model download ───────────────────────────────────────────────────
# In a real scenario you would run:
#   RUN mlflow artifacts download -r ${RUN_ID} -d /app/model
# Here we echo to keep the build self-contained without a live MLflow server.
RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    mkdir -p /app/model && \
    echo "${RUN_ID}" > /app/model/run_id.txt

# ── Copy any inference code you have ─────────────────────────────────────────
# COPY serve.py /app/serve.py

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "-c", \
     "import os; print(f'Model container running. Run ID: {os.environ[\"RUN_ID\"]}')"]