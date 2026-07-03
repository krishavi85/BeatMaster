import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any
import numpy as np
from .config import settings

class AudioProcessingError(RuntimeError):
    pass

def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise AudioProcessingError((result.stderr or result.stdout)[-6000:])
    return result

def probe_audio(path: Path) -> dict[str, Any]:
    result = run([settings.ffprobe_bin, "-v", "error", "-show_entries", "format=duration,size,bit_rate:stream=codec_name,codec_type,sample_rate,channels,channel_layout", "-of", "json", str(path)])
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "audio"), {})
    fmt = payload.get("format", {})
    return {"duration_seconds": float(fmt.get("duration") or 0), "size_bytes": int(fmt.get("size") or path.stat().st_size), "bit_rate": int(fmt.get("bit_rate") or 0), "sample_rate": int(stream.get("sample_rate") or 0), "channels": int(stream.get("channels") or 0), "channel_layout": stream.get("channel_layout"), "codec": stream.get("codec_name")}

def loudness_json(text: str) -> dict[str, Any]:
    for block in reversed(re.findall(r"\{[\s\S]*?\}", text)):
        try:
            data = json.loads(block)
            if "input_i" in data:
                return data
        except json.JSONDecodeError:
            pass
    raise AudioProcessingError("FFmpeg returned no loudness data")

def measure_loudness(path: Path) -> dict[str, Any]:
    result = subprocess.run([settings.ffmpeg_bin, "-hide_banner", "-nostats", "-i", str(path), "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    data = loudness_json(result.stderr)
    return {"integrated_lufs": float(data["input_i"]), "true_peak_dbfs": float(data["input_tp"]), "loudness_range_lu": float(data["input_lra"]), "loudness_threshold_lufs": float(data["input_thresh"])}

def waveform_peaks(audio: np.ndarray, points: int = 1200) -> list[float]:
    chunk = max(1, int(math.ceil(audio.size / points)))
    return [round(float(np.max(np.abs(audio[i:i + chunk]))), 5) for i in range(0, audio.size, chunk)][:points]

def analyze_audio(path: Path) -> dict[str, Any]:
    analysis = {**probe_audio(path), **measure_loudness(path)}
    import librosa
    audio, sample_rate = librosa.load(path, sr=22050, mono=True, duration=600)
    if audio.size:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        rms = max(float(np.sqrt(np.mean(np.square(audio)))), 1e-12)
        peak = max(float(np.max(np.abs(audio))), 1e-12)
        analysis.update({"tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 2), "estimated_key": keys[int(np.argmax(np.mean(chroma, axis=1)))], "rms_dbfs": round(20 * math.log10(rms), 2), "crest_factor_db": round(20 * math.log10(peak) - 20 * math.log10(rms), 2), "spectral_centroid_hz": round(float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))), 2), "waveform_peaks": waveform_peaks(audio)})
    return analysis
