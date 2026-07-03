from pathlib import Path
import subprocess
from typing import Any
from .config import settings
from .audio_probe import AudioProcessingError, loudness_json, measure_loudness, run

def codec_args(fmt: str) -> list[str]:
    if fmt == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", "320k"]
    if fmt == "flac":
        return ["-c:a", "flac", "-compression_level", "8"]
    return ["-c:a", "pcm_s24le"]

def style_filters(style: str) -> list[str]:
    return {
        "transparent": ["highpass=f=25"],
        "warm": ["highpass=f=25", "equalizer=f=180:t=q:w=0.8:g=1.2", "equalizer=f=9000:t=q:w=0.7:g=-0.8"],
        "bright": ["highpass=f=30", "equalizer=f=3500:t=q:w=0.8:g=1.1", "equalizer=f=11000:t=q:w=0.7:g=1.2"],
        "punchy": ["highpass=f=28", "acompressor=threshold=-18dB:ratio=2.2:attack=15:release=120:makeup=1.5"],
        "wide": ["highpass=f=25", "stereotools=mlev=1:slev=1.18:mode=lr>lr"],
    }.get(style, ["highpass=f=25"])

def master_audio(source: Path, output: Path, target_lufs: float, true_peak_db: float, loudness_range: float, style: str) -> dict[str, Any]:
    first = subprocess.run([settings.ffmpeg_bin, "-hide_banner", "-nostats", "-i", str(source), "-af", f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={loudness_range}:print_format=json", "-f", "null", "-"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if first.returncode != 0:
        raise AudioProcessingError(first.stderr[-6000:])
    measured = loudness_json(first.stderr)
    normalized = f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={loudness_range}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true:print_format=json"
    limiter = f"alimiter=limit={10 ** (true_peak_db / 20):.6f}:attack=5:release=50:level=false"
    output.parent.mkdir(parents=True, exist_ok=True)
    run([settings.ffmpeg_bin, "-y", "-hide_banner", "-i", str(source), "-af", ",".join(style_filters(style) + [normalized, limiter]), "-ar", "48000", *codec_args(output.suffix.lstrip(".")), str(output)])
    return {"source_measurement": measured, "final_measurement": measure_loudness(output)}

def mix_audio(inputs: list[tuple[Path, float, float]], output: Path) -> None:
    if not inputs:
        raise AudioProcessingError("At least one unmuted track is required")
    command = [settings.ffmpeg_bin, "-y", "-hide_banner"]
    for path, _, _ in inputs:
        command += ["-i", str(path)]
    chains, labels = [], []
    for index, (_, gain_db, pan) in enumerate(inputs):
        label = f"a{index}"
        chains.append(f"[{index}:a]aresample=48000,volume={gain_db}dB,stereotools=balance_out={pan}[{label}]")
        labels.append(f"[{label}]")
    chains.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:dropout_transition=0,alimiter=limit=0.98:level=false[mix]")
    output.parent.mkdir(parents=True, exist_ok=True)
    run(command + ["-filter_complex", ";".join(chains), "-map", "[mix]", "-ar", "48000", *codec_args(output.suffix.lstrip(".")), str(output)])
