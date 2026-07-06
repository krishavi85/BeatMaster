"""Trainable BeatMaster model family.

The package contains from-scratch reference architectures for an audio codec,
text-conditioned music language model, lyrics language model and singing model.
No pretrained weights are bundled.
"""

from .models import (
    BeatMasterAudioCodec,
    BeatMasterLyricsLM,
    BeatMasterMusicLM,
    BeatMasterSingingModel,
    ModelConfig,
)

__all__ = [
    "BeatMasterAudioCodec",
    "BeatMasterMusicLM",
    "BeatMasterLyricsLM",
    "BeatMasterSingingModel",
    "ModelConfig",
]
