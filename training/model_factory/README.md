# BeatMaster Model Factory

This directory contains a real, from-scratch reference model family for BeatMaster. It does not download or repackage Meta MusicGen weights, cloned singer weights, or undocumented culture models.

## Included model family

1. **BeatMaster Audio Codec**
   - convolutional encoder and decoder;
   - residual vector quantization;
   - discrete audio tokens used by the music model;
   - waveform, spectral and commitment losses.

2. **BeatMaster MusicLM**
   - multilingual text conditioning;
   - autoregressive generation over BeatMaster codec frames;
   - configurable temperature and top-k sampling;
   - long-form generation through model-server chunking and crossfades.

3. **BeatMaster LyricsLM**
   - decoder-only multilingual songwriting model;
   - uses the same transparent Unicode tokenizer;
   - trains only from lyric files listed in the consented dataset manifest.

4. **BeatMaster Singing Model**
   - lyric-frame and MIDI-pitch conditioning;
   - transformer acoustic model;
   - trainable neural waveform generator;
   - requires consented singer recordings and frame-level lyric/pitch alignments.

5. **BeatMaster Model Server**
   - FastAPI generation endpoint;
   - loads your trained codec, MusicLM and tokenizer;
   - connects directly to the main BeatMaster worker through `MUSIC_GENERATION_PROVIDER=beatmaster`.

## Important distinction

The source code is complete enough to prepare data, train checkpoints, package them and serve them. The repository cannot contain already-trained Caribbean or Surinamese weights until you supply a legally usable dataset and GPU training run. Empty or randomly initialized checkpoints are not marketed as working culture models.

## Dataset preparation

The source manifest must use `training/dataset_manifest.schema.json`. Every recording must include:

- explicit machine-learning consent;
- a usable license;
- withdrawal status;
- performers;
- languages;
- region and culture profile;
- train, validation or test split.

Prepare ten-second training segments:

```bash
cd training/model_factory
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD

python scripts/prepare_dataset.py \
  ../my_dataset.json \
  data/prepared \
  --sample-rate 32000 \
  --segment-seconds 10 \
  --overlap-seconds 1
```

The preparation command performs normalization, clipping and silence checks, duplicate fingerprinting, split preservation and consent validation. It produces:

```text
data/prepared/
├── audio/
├── segments.jsonl
└── preparation-report.json
```

## Build the multilingual tokenizer

```bash
python scripts/build_tokenizer.py \
  data/prepared/segments.jsonl \
  data/tokenizer.json \
  --vocabulary-size 16000
```

## Train the audio codec

```bash
python scripts/train_codec.py configs/codec-small.yaml
```

Expected output:

```text
runs/codec-small/codec_latest.pt
```

Train the codec first because MusicLM learns the codec's discrete token vocabulary.

## Produce codec-token training data

```bash
python scripts/tokenize_audio.py \
  data/prepared/segments.jsonl \
  runs/codec-small/codec_latest.pt \
  data/tokenizer.json \
  data/tokens
```

## Train MusicLM

```bash
python scripts/train_music_lm.py configs/music-lm-small.yaml
```

Expected output:

```text
runs/music-lm-small/music_lm_latest.pt
```

## Train LyricsLM

Add `lyrics_path` to the consented recording records, then run:

```bash
python scripts/train_lyrics_lm.py configs/lyrics-small.yaml
```

## Train the singing model

Singing records require `alignment_path`. Each alignment JSON must contain frame-aligned arrays:

```json
{
  "lyric_frame_ids": [12, 12, 12, 25, 25],
  "midi_pitch": [60, 60, 60, 62, 62]
}
```

The frame hop is 320 samples at 32 kHz. Only train voices whose performers explicitly consented to voice-model training and the intended commercial scope.

```bash
python scripts/train_singing.py configs/singing-small.yaml
```

## Package the runtime models

```bash
python scripts/package_models.py \
  --codec runs/codec-small/codec_latest.pt \
  --music runs/music-lm-small/music_lm_latest.pt \
  --tokenizer data/tokenizer.json \
  --output ../../models \
  --name "BeatMaster Suriname Music v1" \
  --license "Your documented model and dataset license"
```

The package command verifies codec compatibility, records provenance and creates SHA-256 checksums.

## Run BeatMaster with your own model

From the repository root:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.models.yml \
  up --build
```

Services:

- BeatMaster application: `http://localhost:8080`
- BeatMaster API: `http://localhost:8000`
- own-model server: `http://localhost:8090`

The override sets:

```env
MUSIC_GENERATION_PROVIDER=beatmaster
BEATMASTER_MODEL_API_URL=http://model-server:8090
```

## Scaling beyond the starter configurations

The starter models are intentionally small enough for validation. A commercially competitive model requires substantially larger hidden dimensions, more layers, a broad consented dataset and distributed GPU training. Increase model capacity only after the data pipeline, validation loss and memorization checks are stable.

Recommended production controls:

- artist-disjoint validation and test splits;
- audio-nearest-neighbor memorization checks;
- community review for every cultural profile;
- dataset and model cards;
- withdrawal and retraining procedures;
- checkpoint signing and immutable provenance;
- generated-audio disclosure and abuse monitoring.
