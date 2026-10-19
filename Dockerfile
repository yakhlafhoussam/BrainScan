# ── BrainScan AI — Dockerfile ────────────────────────────────────────
# Multi-stage-free production image for the Streamlit MRI classifier.
# Base image matches the Python version used to train the model.

FROM python:3.14-slim

# ── System dependencies ─────────────────────────────────────────────
# libgl1 and libglib2.0-0 are required by OpenCV (even the headless build).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ────────────────────────────────────────────────
COPY app.py predict.py ./
COPY models/ models/

# ── Non-root user ────────────────────────────────────────────────────
RUN useradd --create-home appuser
USER appuser

# ── Expose Streamlit port ────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Run the application ─────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
