# BeatMaster

BeatMaster is a self-hosted audio-production workstation. It stores and processes real audio files and never substitutes missing engines with sample values or placeholder output.

## Implemented

- FFprobe metadata and FFmpeg/librosa audio analysis
- Demucs v4 stem separation
- Real multitrack mixing with gain, pan and limiting
- Two-pass EBU R128 mastering
- Optional local MusicGen-compatible generation
- Persistent projects, jobs, progress, errors, previews and downloads
- Runtime capability detection and a responsive dark-purple workspace

## Run

```bash
docker compose up --build
```

Open BeatMaster at `http://localhost:8080` and the API at `http://localhost:8000/docs`.

For GPU processing and local generation:

```bash
cp config.example.env .env
# Set MUSICGEN_MODEL to a compatible model whose license permits your use.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Generation stays disabled unless a model is configured and the AI worker starts. The first Demucs job downloads its selected model weights into the persistent cache.

## Develop

```bash
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.server:application --reload
```

Run the worker in another terminal with `python -m app.worker`.

Production operation still requires compute, disk, backups, HTTPS, authentication, quotas, monitoring, malware scanning and appropriate model/content licenses.
