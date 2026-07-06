from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .models import BeatMasterAudioCodec, BeatMasterMusicLM, ModelConfig
from .tokenizer import BeatMasterTokenizer


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=8, max_length=4000)
    duration_seconds: int = Field(30, ge=4, le=300)
    seed: int | None = None
    temperature: float = Field(1.0, ge=0.1, le=3.0)
    top_k: int = Field(100, ge=1, le=1024)


class Runtime:
    def __init__(self) -> None:
        codec_path = Path(os.environ["BEATMASTER_CODEC_CHECKPOINT"])
        music_path = Path(os.environ["BEATMASTER_MUSIC_CHECKPOINT"])
        tokenizer_path = Path(os.environ["BEATMASTER_TOKENIZER"])
        requested = os.getenv("BEATMASTER_MODEL_DEVICE", "auto")
        self.device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
        codec_payload = torch.load(codec_path, map_location=self.device, weights_only=False)
        music_payload = torch.load(music_path, map_location=self.device, weights_only=False)
        codec_config = ModelConfig.from_dict(codec_payload["config"])
        music_config = ModelConfig.from_dict(music_payload["config"])
        if codec_config.codec_bins != music_config.codec_bins or codec_config.codec_codebooks != music_config.codec_codebooks:
            raise RuntimeError("Codec and MusicLM checkpoints use incompatible token settings")
        self.config = music_config
        self.codec = BeatMasterAudioCodec(codec_config).to(self.device)
        self.codec.load_state_dict(codec_payload["model"])
        self.codec.eval()
        self.music = BeatMasterMusicLM(music_config).to(self.device)
        self.music.load_state_dict(music_payload["model"])
        self.music.eval()
        self.tokenizer = BeatMasterTokenizer.load(tokenizer_path)
        self.sample_rate = codec_config.sample_rate
        self.hop_length = self.codec.hop_length
        self.model_names = {"codec": codec_path.name, "music": music_path.name, "tokenizer": tokenizer_path.name}

    @staticmethod
    def crossfade(parts: list[np.ndarray], samples: int) -> np.ndarray:
        if not parts:
            raise RuntimeError("No generated parts")
        output = parts[0]
        for part in parts[1:]:
            overlap = min(samples, len(output) // 4, len(part) // 4)
            if overlap <= 0:
                output = np.concatenate([output, part])
                continue
            fade = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
            mixed = output[-overlap:] * (1.0 - fade) + part[:overlap] * fade
            output = np.concatenate([output[:-overlap], mixed, part[overlap:]])
        return output

    def generate(self, request: GenerationRequest) -> np.ndarray:
        text_ids = self.tokenizer.encode(request.prompt, maximum_length=self.config.text_max_length)
        text = torch.tensor([text_ids], dtype=torch.long, device=self.device)
        total_frames = int(math.ceil(request.duration_seconds * self.sample_rate / self.hop_length))
        maximum_frames = min(self.config.max_audio_frames - 1, int(os.getenv("BEATMASTER_GENERATION_CHUNK_FRAMES", "2048")))
        parts: list[np.ndarray] = []
        remaining = total_frames
        index = 0
        with torch.inference_mode():
            while remaining > 0:
                frames = min(maximum_frames, remaining)
                codes = self.music.generate(text, frames, temperature=request.temperature, top_k=request.top_k, seed=None if request.seed is None else request.seed + index)
                waveform = self.codec.decode(codes)[0].mean(dim=0).detach().cpu().float().numpy()
                parts.append(waveform)
                remaining -= frames
                index += 1
        result = self.crossfade(parts, int(self.sample_rate * 0.5))
        target_samples = request.duration_seconds * self.sample_rate
        result = result[:target_samples]
        peak = max(float(np.max(np.abs(result))), 1e-8)
        return np.asarray(result / peak * 0.95, dtype=np.float32)


@lru_cache(maxsize=1)
def runtime() -> Runtime:
    return Runtime()


application = FastAPI(title="BeatMaster Model Server", version="1.0.0")


@application.get("/health")
def health():
    try:
        value = runtime()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "device": str(value.device), "sample_rate": value.sample_rate, "models": value.model_names}


@application.post("/generate")
def generate(request: GenerationRequest):
    try:
        waveform = runtime().generate(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    buffer = io.BytesIO()
    sf.write(buffer, waveform, runtime().sample_rate, format="WAV", subtype="PCM_16")
    return Response(buffer.getvalue(), media_type="audio/wav", headers={"X-BeatMaster-Model": runtime().model_names["music"]})
