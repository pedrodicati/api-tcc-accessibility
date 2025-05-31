ARG PYTHON_VERSION=3.10.12
FROM python:${PYTHON_VERSION}-slim

# Create a non-root user and group
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# enable use gpu
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set WORKDIR early, it will be created as root
WORKDIR /app

# Copy requirements.txt and ensure it's owned by appuser later (or now, but root installs)
COPY ./requirements.txt ./requirements.txt

RUN apt update && apt install -y ffmpeg

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code into the container, ensure appuser owns it.
# WORKDIR is already /app
COPY --chown=appuser:appgroup ./app ./

# Ensure the /app directory itself and its contents (like requirements.txt if copied earlier without chown,
# and the log file that will be created) are writable by appuser.
# Note: pip installed packages are global and readable. We are concerned with app files and runtime generated files.
# Since WORKDIR /app was created by root, we need to chown it.
# If requirements.txt was copied to /app/requirements.txt, this chown will cover it.
# The COPY --chown for ./app ./ already sets app code ownership.
# This chown makes sure /app itself is writable for log.log, and requirements.txt if it's in /app.
RUN chown appuser:appgroup /app && chown appuser:appgroup ./requirements.txt

USER appuser

# Port needs to be > 1024 if running as non-root, but Docker maps it.
# The internal port 8000 is fine.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on", "--reload"]
