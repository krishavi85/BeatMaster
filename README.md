# BeatMaster

BeatMaster is a self-hosted multilingual AI music-production studio for artists, songwriters and producers. It stores and processes real project assets and does not replace unavailable engines with demonstration values, fabricated meters or placeholder audio.

## Working production features

- Audio upload, FFprobe metadata and FFmpeg/librosa analysis
- Demucs v4 source separation with editable stem mixing
- Per-track gain, pan and mute controls
- Two-pass EBU R128 mastering
- Chord detection with downloadable JSON, text and timeline files
- Audio-to-MIDI transcription with a real `.mid` file
- Multilingual AI songwriting through Ollama, an OpenAI-compatible endpoint or a configured local Transformers model
- Optional singing-synthesis integration through a real REST provider
- Culture-aware music generation using transparent Caribbean and Surinamese production profiles
- Optional routing to administrator-registered culture-specific fine-tuned models
- Complete production packages containing a music bed, stems, MIDI, chords, lyrics and DAW assets
- REAPER `.rpp` projects and generic aligned-stem interchange ZIP files
- Persistent projects, jobs, progress, failures, previews and downloads
- Runtime capability reporting that disables unavailable tools instead of pretending they work

## Cultural profiles

BeatMaster includes explicit profiles for:

- Surinamese Kaseko
- Surinamese Kawina
- Surinamese Baithak Gana
- Soca
- Calypso
- Reggae
- Dancehall
- Zouk / Kompa
- Chutney / Chutney Soca

These built-in profiles provide inspectable tempo, rhythm, instrumentation and production guidance. They are **not falsely presented as trained culture models**. To use genuinely fine-tuned weights, register them with `CULTURE_MODEL_MAP`. See [`docs/CULTURAL_MODELS.md`](docs/CULTURAL_MODELS.md).

## Run the CPU workstation

```bash
docker compose up --build
```

Open:

- BeatMaster: `http://localhost:8080`
- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Runtime capabilities: `http://localhost:8000/api/capabilities`
- Culture profiles: `http://localhost:8000/api/culture-profiles`

The CPU deployment provides upload, analysis, Demucs separation, mixing, mastering, chords, MIDI and DAW export. The first Demucs job downloads its selected model weights into the persistent cache.

## Enable local music generation

```bash
cp config.env.sample .env
```

Set a compatible model whose license permits your use:

```env
ENABLE_LOCAL_MUSICGEN=true
MUSICGEN_MODEL=your-compatible-model-id
```

For CUDA processing:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Connect multilingual songwriting

### Ollama

```env
LYRICS_PROVIDER=ollama
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_LYRICS_MODEL=your-text-model
```

### OpenAI-compatible endpoint

```env
LYRICS_PROVIDER=openai-compatible
LYRICS_API_URL=https://your-provider.example/v1/chat/completions
LYRICS_API_KEY=your-token
LYRICS_MODEL=your-model
```

### Local Transformers model

```env
LYRICS_PROVIDER=local-transformers
LYRICS_LOCAL_MODEL=your-text-generation-model
```

## Connect singing synthesis

BeatMaster accepts a provider that receives a multipart POST containing:

- `lyrics`: UTF-8 text file
- `midi`: optional Standard MIDI File
- `language`
- `title`
- `voice_id`
- `output_format=wav`

The service must return audio bytes directly or JSON containing `audio_url`.

```env
SINGING_PROVIDER=rest
SINGING_API_URL=https://your-singing-service.example/render
SINGING_API_KEY=your-token
SINGING_VOICE_ID=your-default-voice
```

If the provider is unavailable, BeatMaster reports it and disables vocal rendering.

## Register fine-tuned culture models

```env
CULTURE_MODEL_MAP={"suriname-kaseko":"your-org/kaseko-model","caribbean-soca":"your-org/soca-model"}
```

Every generated asset records the exact model ID, culture profile, enhanced prompt and whether a mapped fine-tuned model or base-model prompt conditioning was used.

## Development

```bash
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.server:application --reload
```

Run the worker in another terminal:

```bash
cd backend
. .venv/bin/activate
python -m app.worker
```

Run verification:

```bash
cd backend
pytest -q
python -m compileall -q app tests
```

## Deployment responsibilities

Production operation still requires appropriate CPU/GPU capacity, persistent storage, backups, HTTPS, authentication, authorization, quotas, monitoring, malware scanning and legally valid model/content licenses. Culture-specific model weights require consented datasets, community review and documented provenance; BeatMaster does not invent those weights or claim authenticity without evidence.
