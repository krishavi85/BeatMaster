# BeatMaster

BeatMaster is a self-hosted audio-production workstation with real file processing.

## Working processors

- Audio upload and FFprobe metadata
- Demucs v4 source separation
- FFmpeg multitrack mixing
- Two-pass EBU R128 mastering
- Loudness, tempo, key, waveform and technical analysis
- Optional local MusicGen generation on a configured AI worker
- Persistent processing jobs, real progress, audio preview and downloads

## Start

```bash
cp .env.example .env
docker compose up --build
```

Open the web application at `http://localhost:8080` and API documentation at `http://localhost:8000/docs`.

For a CUDA worker and local MusicGen:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The application never substitutes missing processors with sample values or placeholder audio. When a runtime is unavailable, the capability screen reports it and the related action remains disabled.

## Development

```bash
cd web && npm install && npm run dev
cd backend && pip install -r requirements-worker.txt
uvicorn app.main:app --reload
python -m app.worker
```

MusicGen model licensing must be reviewed before commercial deployment. Production hosting also needs authentication, TLS, quotas, backups, monitoring and sufficient CPU/GPU capacity.
