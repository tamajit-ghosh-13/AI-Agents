# ─────────────────────────────────────────────
# ShortlistIQ — Redrob AI Ranker
# Base: slim Python 3.11 on Debian Bookworm
# ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# Install system packages:
#   git         — needed to clone the repo and run git lfs
#   git-lfs     — to pull candidate_embeddings.npy from LFS
#   curl + ca-certificates — required by the git-lfs installer script
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates && \
    # Official Git LFS installer
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install && \
    # Clean up to keep the image lean
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# ── Copy only dependency files first (layer-cache friendly) ──
COPY requirements.txt .

# Install Python dependencies (no cache to keep image size down)
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy the rest of the source code ──
COPY . .

# Default command — run the ranker on the sample file.
# Override at runtime by passing your own arguments:
#   docker run ... python rank.py --input your_file.jsonl ...
CMD ["python", "rank.py", \
     "--input",  "candidates_sample.jsonl", \
     "--output", "submission_sample.csv", \
     "--jd",     "job_description.txt"]
