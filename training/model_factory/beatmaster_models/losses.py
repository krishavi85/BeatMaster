import torch
import torch.nn.functional as F


def stft_loss(prediction, target):
    prediction = prediction.mean(dim=1)
    target = target.mean(dim=1)
    total = prediction.new_zeros(())
    for size, hop in ((512, 128), (1024, 256), (2048, 512)):
        window = torch.hann_window(size, device=prediction.device, dtype=prediction.dtype)
        predicted = torch.stft(prediction, size, hop, size, window, return_complex=True).abs().clamp_min(1e-7)
        expected = torch.stft(target, size, hop, size, window, return_complex=True).abs().clamp_min(1e-7)
        total = total + torch.linalg.vector_norm(expected - predicted) / torch.linalg.vector_norm(expected).clamp_min(1e-7)
        total = total + F.l1_loss(predicted.log(), expected.log())
    return total / 3


def codec_loss(output, target):
    waveform = F.l1_loss(output["waveform"], target)
    spectral = stft_loss(output["waveform"], target)
    commitment = output["commitment_loss"]
    return {"loss": waveform + spectral + commitment, "waveform": waveform, "spectral": spectral, "commitment": commitment}


def music_loss(logits, codes):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.reshape(-1))


def text_loss(logits, tokens, pad_id=0):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=pad_id)


def singing_loss(output, target_mel, target_waveform=None):
    frames = min(output["mel"].shape[-1], target_mel.shape[-1])
    mel = F.l1_loss(output["mel"][..., :frames], target_mel[..., :frames])
    wave = mel.new_zeros(())
    if target_waveform is not None:
        samples = min(output["waveform"].shape[-1], target_waveform.shape[-1])
        wave = stft_loss(output["waveform"][..., :samples], target_waveform[..., :samples])
    return {"loss": mel + wave, "mel": mel, "waveform": wave}
