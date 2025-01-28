ARG PYTHON_VERSION=3.10.12
FROM python:${PYTHON_VERSION}-slim

# enable use gpu
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY ./requirements.txt ./

RUN apt update && apt install -y ffmpeg

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on", "--reload"]