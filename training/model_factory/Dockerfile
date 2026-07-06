FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/beatmaster-model
COPY training/model_factory/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY training/model_factory/beatmaster_models ./beatmaster_models

EXPOSE 8090
CMD ["uvicorn", "beatmaster_models.server:application", "--host", "0.0.0.0", "--port", "8090"]
