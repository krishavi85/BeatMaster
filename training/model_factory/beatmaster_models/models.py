from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    sample_rate: int = 32000
    audio_channels: int = 1
    codec_hidden: int = 128
    codec_latent: int = 256
    codec_codebooks: int = 4
    codec_bins: int = 1024
    codec_strides: tuple[int, ...] = (2, 4, 5, 8)
    text_vocab_size: int = 16000
    text_max_length: int = 256
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 8
    transformer_ff: int = 2048
    dropout: float = 0.1
    max_audio_frames: int = 4096
    n_mels: int = 100
    singing_hidden: int = 384
    singing_layers: int = 6
    singing_heads: int = 6
    vocoder_upsample: tuple[int, ...] = (8, 8, 5)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["codec_strides"] = list(self.codec_strides)
        data["vocoder_upsample"] = list(self.vocoder_upsample)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        values = dict(data)
        if "codec_strides" in values:
            values["codec_strides"] = tuple(values["codec_strides"])
        if "vocoder_upsample" in values:
            values["vocoder_upsample"] = tuple(values["vocoder_upsample"])
        return cls(**values)


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(value)


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with straight-through gradients.

    The implementation is deliberately self-contained so BeatMaster checkpoints do
    not depend on third-party pretrained codec weights.
    """

    def __init__(self, dimension: int, codebooks: int, bins: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.codebooks = codebooks
        self.bins = bins
        self.embeddings = nn.ModuleList([nn.Embedding(bins, dimension) for _ in range(codebooks)])
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -1.0 / bins, 1.0 / bins)

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # latents: [batch, dimension, frames]
        residual = latents.transpose(1, 2).contiguous()
        quantized_sum = torch.zeros_like(residual)
        codes: list[Tensor] = []
        commitment = latents.new_zeros(())
        for embedding in self.embeddings:
            flat = residual.reshape(-1, self.dimension)
            weight = embedding.weight
            distances = (
                flat.square().sum(dim=1, keepdim=True)
                - 2.0 * flat @ weight.t()
                + weight.square().sum(dim=1).unsqueeze(0)
            )
            indices = distances.argmin(dim=1).view(residual.shape[0], residual.shape[1])
            quantized = embedding(indices)
            commitment = commitment + F.mse_loss(residual, quantized.detach()) + 0.25 * F.mse_loss(quantized, residual.detach())
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            codes.append(indices)
        straight_through = latents.transpose(1, 2) + (quantized_sum - latents.transpose(1, 2)).detach()
        return straight_through.transpose(1, 2).contiguous(), torch.stack(codes, dim=1), commitment / self.codebooks

    def decode(self, codes: Tensor) -> Tensor:
        # codes: [batch, codebooks, frames]
        if codes.ndim != 3 or codes.shape[1] != self.codebooks:
            raise ValueError(f"Expected [batch, {self.codebooks}, frames] codes")
        value = None
        for index, embedding in enumerate(self.embeddings):
            decoded = embedding(codes[:, index])
            value = decoded if value is None else value + decoded
        assert value is not None
        return value.transpose(1, 2).contiguous()


class BeatMasterAudioCodec(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.codec_hidden
        encoder: list[nn.Module] = [nn.Conv1d(config.audio_channels, channels, 7, padding=3)]
        for index, stride in enumerate(config.codec_strides):
            encoder.extend([
                ResidualBlock1d(channels, dilation=1),
                ResidualBlock1d(channels, dilation=3),
                nn.Conv1d(channels, channels * 2, kernel_size=2 * stride, stride=stride, padding=stride // 2),
            ])
            channels *= 2
        encoder.append(nn.Conv1d(channels, config.codec_latent, 3, padding=1))
        self.encoder = nn.Sequential(*encoder)
        self.quantizer = ResidualVectorQuantizer(config.codec_latent, config.codec_codebooks, config.codec_bins)
        decoder: list[nn.Module] = [nn.Conv1d(config.codec_latent, channels, 3, padding=1)]
        for stride in reversed(config.codec_strides):
            decoder.extend([
                nn.ConvTranspose1d(channels, channels // 2, kernel_size=2 * stride, stride=stride, padding=stride // 2),
                ResidualBlock1d(channels // 2, dilation=1),
                ResidualBlock1d(channels // 2, dilation=3),
            ])
            channels //= 2
        decoder.extend([nn.GroupNorm(1, channels), nn.SiLU(), nn.Conv1d(channels, config.audio_channels, 7, padding=3), nn.Tanh()])
        self.decoder = nn.Sequential(*decoder)

    @property
    def hop_length(self) -> int:
        return math.prod(self.config.codec_strides)

    def encode(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encoder(waveform)
        quantized, codes, commitment = self.quantizer(latents)
        return codes, commitment

    def decode(self, codes: Tensor) -> Tensor:
        return self.decoder(self.quantizer.decode(codes))

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        latents = self.encoder(waveform)
        quantized, codes, commitment = self.quantizer(latents)
        reconstruction = self.decoder(quantized)
        target_length = waveform.shape[-1]
        if reconstruction.shape[-1] > target_length:
            reconstruction = reconstruction[..., :target_length]
        elif reconstruction.shape[-1] < target_length:
            reconstruction = F.pad(reconstruction, (0, target_length - reconstruction.shape[-1]))
        return {"waveform": reconstruction, "codes": codes, "commitment_loss": commitment}


class PositionalEmbedding(nn.Module):
    def __init__(self, maximum: int, dimension: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(maximum, dimension)

    def forward(self, value: Tensor) -> Tensor:
        positions = torch.arange(value.shape[1], device=value.device).unsqueeze(0)
        return value + self.embedding(positions)


class BeatMasterMusicLM(nn.Module):
    """Text-conditioned autoregressive model over BeatMaster codec frames."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.transformer_dim
        self.text_embedding = nn.Embedding(config.text_vocab_size, dim, padding_idx=0)
        self.text_position = PositionalEmbedding(config.text_max_length, dim)
        text_layer = nn.TransformerEncoderLayer(dim, config.transformer_heads, config.transformer_ff, config.dropout, batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(text_layer, max(2, config.transformer_layers // 2))
        self.code_embeddings = nn.ModuleList([nn.Embedding(config.codec_bins + 1, dim) for _ in range(config.codec_codebooks)])
        self.audio_position = PositionalEmbedding(config.max_audio_frames, dim)
        decoder_layer = nn.TransformerDecoderLayer(dim, config.transformer_heads, config.transformer_ff, config.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.transformer_layers)
        self.heads = nn.ModuleList([nn.Linear(dim, config.codec_bins) for _ in range(config.codec_codebooks)])
        self.start_token = config.codec_bins

    def encode_text(self, text_ids: Tensor, text_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        value = self.text_position(self.text_embedding(text_ids))
        key_padding = ~text_mask.bool() if text_mask is not None else text_ids.eq(0)
        return self.text_encoder(value, src_key_padding_mask=key_padding), key_padding

    def _audio_embedding(self, codes: Tensor) -> Tensor:
        # codes [batch, codebooks, frames]
        embeddings = [self.code_embeddings[index](codes[:, index]) for index in range(self.config.codec_codebooks)]
        return self.audio_position(torch.stack(embeddings, dim=0).sum(dim=0) / math.sqrt(self.config.codec_codebooks))

    def forward(self, text_ids: Tensor, audio_codes: Tensor, text_mask: Tensor | None = None) -> Tensor:
        if audio_codes.shape[-1] < 2:
            raise ValueError("At least two audio frames are required")
        memory, memory_padding = self.encode_text(text_ids, text_mask)
        start = torch.full((audio_codes.shape[0], audio_codes.shape[1], 1), self.start_token, device=audio_codes.device, dtype=audio_codes.dtype)
        decoder_codes = torch.cat([start, audio_codes[:, :, :-1]], dim=-1)
        target = self._audio_embedding(decoder_codes)
        length = target.shape[1]
        causal_mask = torch.full((length, length), float("-inf"), device=target.device).triu(1)
        hidden = self.decoder(target, memory, tgt_mask=causal_mask, memory_key_padding_mask=memory_padding)
        return torch.stack([head(hidden) for head in self.heads], dim=1)

    @torch.inference_mode()
    def generate(self, text_ids: Tensor, frames: int, *, temperature: float = 1.0, top_k: int = 100, seed: int | None = None) -> Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        memory, memory_padding = self.encode_text(text_ids)
        codes = torch.empty((text_ids.shape[0], self.config.codec_codebooks, 0), device=text_ids.device, dtype=torch.long)
        for _ in range(frames):
            start = torch.full((codes.shape[0], codes.shape[1], 1), self.start_token, device=codes.device, dtype=torch.long)
            decoder_codes = start if codes.shape[-1] == 0 else torch.cat([start, codes], dim=-1)
            target = self._audio_embedding(decoder_codes)
            length = target.shape[1]
            causal_mask = torch.full((length, length), float("-inf"), device=target.device).triu(1)
            hidden = self.decoder(target, memory, tgt_mask=causal_mask, memory_key_padding_mask=memory_padding)[:, -1]
            frame_codes: list[Tensor] = []
            for head in self.heads:
                logits = head(hidden) / max(temperature, 1e-4)
                if 0 < top_k < logits.shape[-1]:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                frame_codes.append(torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1))
            next_frame = torch.stack(frame_codes, dim=1).unsqueeze(-1)
            codes = torch.cat([codes, next_frame], dim=-1)
        return codes


class BeatMasterLyricsLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.transformer_dim
        self.embedding = nn.Embedding(config.text_vocab_size, dim, padding_idx=0)
        self.position = PositionalEmbedding(config.text_max_length, dim)
        layer = nn.TransformerEncoderLayer(dim, config.transformer_heads, config.transformer_ff, config.dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.transformer_layers)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, config.text_vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        value = self.position(self.embedding(token_ids))
        length = token_ids.shape[1]
        causal_mask = torch.full((length, length), float("-inf"), device=token_ids.device).triu(1)
        padding = token_ids.eq(0)
        hidden = self.transformer(value, mask=causal_mask, src_key_padding_mask=padding)
        return self.output(self.norm(hidden))

    @torch.inference_mode()
    def generate(self, prefix: Tensor, maximum_tokens: int = 512, temperature: float = 0.9, eos_id: int = 3) -> Tensor:
        tokens = prefix
        for _ in range(maximum_tokens):
            window = tokens[:, -self.config.text_max_length :]
            logits = self(window)[:, -1] / max(temperature, 1e-4)
            next_token = torch.multinomial(logits.softmax(dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if bool((next_token == eos_id).all()):
                break
        return tokens


class SingingAcousticModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.singing_hidden
        self.token_embedding = nn.Embedding(config.text_vocab_size, hidden, padding_idx=0)
        self.pitch_embedding = nn.Embedding(129, hidden, padding_idx=0)
        self.position = PositionalEmbedding(config.max_audio_frames, hidden)
        layer = nn.TransformerEncoderLayer(hidden, config.singing_heads, hidden * 4, config.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.singing_layers)
        self.mel_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, config.n_mels))

    def forward(self, lyric_frame_ids: Tensor, midi_pitch: Tensor) -> Tensor:
        midi_pitch = midi_pitch.clamp(0, 128)
        value = self.position(self.token_embedding(lyric_frame_ids) + self.pitch_embedding(midi_pitch))
        hidden = self.encoder(value, src_key_padding_mask=lyric_frame_ids.eq(0))
        return self.mel_head(hidden).transpose(1, 2)


class BeatMasterVocoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = 512
        layers: list[nn.Module] = [nn.Conv1d(config.n_mels, channels, 7, padding=3)]
        for stride in config.vocoder_upsample:
            next_channels = max(32, channels // 2)
            layers.extend([
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(channels, next_channels, kernel_size=stride * 2, stride=stride, padding=stride // 2),
                ResidualBlock1d(next_channels, dilation=1),
                ResidualBlock1d(next_channels, dilation=3),
            ])
            channels = next_channels
        layers.extend([nn.LeakyReLU(0.2), nn.Conv1d(channels, 1, 7, padding=3), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, mel: Tensor) -> Tensor:
        return self.net(mel)


class BeatMasterSingingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.acoustic = SingingAcousticModel(config)
        self.vocoder = BeatMasterVocoder(config)

    def forward(self, lyric_frame_ids: Tensor, midi_pitch: Tensor) -> dict[str, Tensor]:
        mel = self.acoustic(lyric_frame_ids, midi_pitch)
        waveform = self.vocoder(mel)
        return {"mel": mel, "waveform": waveform}
