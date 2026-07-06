# Run BeatMaster with your trained model

After training, package the codec and music checkpoints:

```bash
cd training/model_factory
export PYTHONPATH=$PWD
python scripts/package_models.py \
  --codec runs/codec-small/codec_latest.pt \
  --music runs/music-lm-small/music_lm_latest.pt \
  --tokenizer data/tokenizer.json \
  --output ../../models \
  --name "BeatMaster Suriname Music v1" \
  --license "Your documented model and dataset license"
```

From the repository root, start the application and model server:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.own-model.yml \
  up --build
```

Open:

- BeatMaster: `http://localhost:8080`
- API: `http://localhost:8000/docs`
- model-server health: `http://localhost:8090/health`

The stack configures `MUSIC_GENERATION_PROVIDER=beatmaster`, points the worker to the model server, and satisfies the API's generation-runtime validation. The server will fail clearly when the packaged checkpoint files are absent or incompatible.
