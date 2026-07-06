from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from beatmaster_models.data import prepare_dataset
from beatmaster_models.models import BeatMasterAudioCodec, BeatMasterLyricsLM, BeatMasterMusicLM, BeatMasterSingingModel, ModelConfig
from beatmaster_models.tokenizer import BeatMasterTokenizer


def tiny_config():
    return ModelConfig(
        sample_rate=8000,
        codec_hidden=8,
        codec_latent=16,
        codec_codebooks=2,
        codec_bins=32,
        codec_strides=(2, 2),
        text_vocab_size=32,
        text_max_length=32,
        transformer_dim=32,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff=64,
        max_audio_frames=64,
        n_mels=16,
        singing_hidden=32,
        singing_layers=2,
        singing_heads=4,
        vocoder_upsample=(2, 2),
    )


def test_codec_round_trip_shapes():
    config = tiny_config()
    model = BeatMasterAudioCodec(config)
    waveform = torch.randn(2, 1, 256)
    output = model(waveform)
    assert output["waveform"].shape == waveform.shape
    assert output["codes"].shape[1] == config.codec_codebooks
    decoded = model.decode(output["codes"])
    assert decoded.shape[0] == waveform.shape[0]


def test_music_and_text_models_forward():
    config = tiny_config()
    music = BeatMasterMusicLM(config)
    text_ids = torch.randint(1, config.text_vocab_size, (2, 8))
    codes = torch.randint(0, config.codec_bins, (2, config.codec_codebooks, 12))
    logits = music(text_ids, codes)
    assert logits.shape == (2, config.codec_codebooks, 12, config.codec_bins)
    lyrics = BeatMasterLyricsLM(config)
    text_logits = lyrics(text_ids)
    assert text_logits.shape == (2, 8, config.text_vocab_size)


def test_singing_model_forward():
    config = tiny_config()
    model = BeatMasterSingingModel(config)
    tokens = torch.randint(1, config.text_vocab_size, (2, 12))
    pitch = torch.randint(1, 100, (2, 12))
    output = model(tokens, pitch)
    assert output["mel"].shape == (2, config.n_mels, 12)
    assert output["waveform"].shape[0] == 2


def test_unicode_tokenizer_round_trip(tmp_path: Path):
    tokenizer = BeatMasterTokenizer.train(["Mi lobi yu", "Switi nyan", "हम नहीं"], vocabulary_size=64, minimum_frequency=1)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BeatMasterTokenizer.load(path)
    ids = loaded.encode("Mi lobi yu")
    assert loaded.decode(ids) == "Mi lobi yu"


def test_dataset_preparation_enforces_consent(tmp_path: Path):
    sample_rate = 8000
    time = np.linspace(0, 2, sample_rate * 2, endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * 220 * time)
    audio = tmp_path / "song.wav"
    sf.write(audio, waveform, sample_rate)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"dataset_name":"test","version":"1","profile_id":"suriname-kaseko","community_reviewers":[{"name":"Reviewer","role":"musician","community_affiliation":"test"}],"recordings":[{"id":"song","path":"song.wav","license":"owned","ml_training_consent":true,"withdrawn":false,"region":"Suriname","languages":["srn"],"performers":["consenting performer"],"split":"train"}]}',
        encoding="utf-8",
    )
    report = prepare_dataset(manifest, tmp_path / "prepared", sample_rate=sample_rate, segment_seconds=1.0, overlap_seconds=0.0)
    assert report["segment_count"] == 2
