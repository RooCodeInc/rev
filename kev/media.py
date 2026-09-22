"""Media references -> what the base's processor takes (kev.model.prepare_media).

A request or record carries media as references: {"type": "image" | "audio" | "video", and one of "path" (a local file;
training data and benchmarks only, refused by the server), "url", or "data" (base64)}. load() turns one into
{"type", "data"}: a PIL image, a mono float32 waveform at 16 kHz, or a uint8 array of sampled video frames [F, H, W, 3]
with the metadata the processor needs to timestamp them. Decoding needs the `media` extra (Pillow, PyAV).
"""
import base64
import functools
import io
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000        # Gemma 4's audio feature extractor and the 12B's raw-sample embedder both take 16 kHz mono
VIDEO_FRAMES = 8           # frames sampled per video by default: about 70 tokens each (plus a timestamp) on Gemma 4
MAX_URL_BYTES = 64 << 20


def _bytes(ref):
    if ref.get("data") is not None: return base64.b64decode(ref["data"])
    if ref.get("url") is not None:
        with urllib.request.urlopen(ref["url"], timeout=30) as r:
            body = r.read(MAX_URL_BYTES + 1)
        if len(body) > MAX_URL_BYTES: raise ValueError(f"media at {ref['url']} is over {MAX_URL_BYTES >> 20} MB")
        return body
    if ref.get("path") is not None: return Path(ref["path"]).read_bytes()
    raise ValueError("a media item needs one of path, url or data")


def load_image(raw):
    from PIL import Image
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_audio(raw):
    """Any container/codec PyAV reads -> mono float32 at SAMPLE_RATE."""
    import av
    out = []
    with av.open(io.BytesIO(raw)) as f:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in f.decode(audio=0):
            out += [r.to_ndarray().reshape(-1) for r in resampler.resample(frame)]
        out += [r.to_ndarray().reshape(-1) for r in resampler.resample(None)]
    if not out: raise ValueError("no audio decoded")
    return np.concatenate(out).astype(np.float32)


def load_video(raw, frames=VIDEO_FRAMES):
    """`frames` frames evenly spaced over the video, and metadata that makes the processor keep all of them and timestamp
    them at their real times (it samples at `fps` over `total_num_frames`: the effective rate of the kept frames)."""
    import av
    with av.open(io.BytesIO(raw)) as f:
        decoded = [fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)]
        stream = f.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else len(decoded) / float(stream.average_rate or 24)
    if not decoded: raise ValueError("no video frames decoded")
    keep = np.linspace(0, len(decoded) - 1, min(frames, len(decoded))).round().astype(int)
    fps = len(keep) / duration if duration > 0 else 1.0
    return np.stack([decoded[i] for i in keep]), {"fps": fps, "total_num_frames": len(keep)}


@functools.lru_cache(maxsize=4096)
def _load_path(kind, path):
    return _decode(kind, Path(path).read_bytes())


def _decode(kind, raw):
    if kind == "image": return {"type": "image", "data": load_image(raw)}
    if kind == "audio": return {"type": "audio", "data": load_audio(raw)}
    if kind == "video":
        frames, meta = load_video(raw)
        return {"type": "video", "data": frames, "metadata": meta, "num_frames": len(frames)}
    raise ValueError(f"media type must be image, audio or video, got {kind!r}")


def load(ref):
    """One media reference -> the item prepare_media takes. Local files are decoded once per process (training
    re-encodes every record every epoch)."""
    if ref.get("path") is not None and ref.get("data") is None and ref.get("url") is None:
        return _load_path(ref["type"], str(Path(ref["path"]).resolve()))
    return _decode(ref["type"], _bytes(ref))
