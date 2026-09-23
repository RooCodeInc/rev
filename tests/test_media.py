"""Media references end to end without model weights: decoding (image, audio resampled to 16 kHz mono, video frames),
the API's Media field, records carrying media through materialize, and the server refusing local paths.
Needs the media extra (Pillow, PyAV); skipped without it.
Run: uv run --extra serve --extra media python -m pytest tests/test_media.py -q
"""
import base64, io, json, wave

import numpy as np
import pytest

pytest.importorskip("av"); pytest.importorskip("PIL")


def wav_bytes(seconds=0.5, rate=8000, freq=440.0):
    t = np.arange(int(seconds * rate)) / rate
    pcm = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    return buf.getvalue()


def png_bytes(w=40, h=30):
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (w, h), (200, 30, 30)).save(buf, "PNG"); return buf.getvalue()


def mp4_bytes(frames=12, fps=6):
    import av
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as f:
        s = f.add_stream("mpeg4", rate=fps); s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            img = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")): f.mux(p)
        for p in s.encode(): f.mux(p)
    return buf.getvalue()


def test_load_decodes_every_kind(tmp_path):
    from kev.media import SAMPLE_RATE, load
    (tmp_path / "a.wav").write_bytes(wav_bytes(0.5, 8000))
    audio = load({"type": "audio", "path": str(tmp_path / "a.wav")})["data"]
    assert audio.dtype == np.float32 and audio.ndim == 1 and abs(len(audio) - SAMPLE_RATE // 2) < 400 and 0.3 < np.abs(audio).max() <= 1.0
    image = load({"type": "image", "data": base64.b64encode(png_bytes()).decode()})["data"]
    assert image.size == (40, 30) and image.mode == "RGB"
    video = load({"type": "video", "data": base64.b64encode(mp4_bytes()).decode()})
    assert video["data"].shape[0] == 8 and video["data"].shape[-1] == 3 and video["num_frames"] == 8
    assert abs(video["metadata"]["fps"] - 8 / 2.0) < 0.5                       # 8 of 12 frames kept over 2 s
    with pytest.raises(ValueError): load({"type": "image"})


def test_api_media_field_and_records(tmp_path):
    from kev.api import SystemOneRequest, to_record
    from kev.data import load_records, materialize
    q = {"q": {"type": "noul", "instructions": "Is it red?"}}
    rec, _ = to_record(SystemOneRequest.model_validate({"state": "s", "questions": q, "media": [{"type": "image", "data": "eA=="}]}))
    assert rec["media"] == [{"type": "image", "data": "eA=="}]
    assert "media" not in to_record(SystemOneRequest.model_validate({"state": "s", "questions": q}))[0]
    for bad in ({"type": "image"}, {"type": "image", "data": "x", "url": "http://e"}, {"type": "gif", "data": "x"}):
        with pytest.raises(ValueError): SystemOneRequest.model_validate({"state": "s", "questions": q, "media": [bad]})
    (tmp_path / "m").mkdir(); (tmp_path / "m" / "x.png").write_bytes(png_bytes())
    line = json.dumps({"state": "", "media": [{"type": "image", "path": "m/x.png"}], "questions": {"red": {"type": "noul", "instructions": "Red?", "label": True}}})
    (tmp_path / "d.jsonl").write_text(line + "\n", encoding="utf-8")
    r = load_records(tmp_path / "d.jsonl")[0]
    assert r["media"][0]["path"] == str((tmp_path / "m" / "x.png").resolve())   # relative to the JSONL file
    assert materialize(r)["media"] == r["media"]


def test_server_refuses_local_paths_and_urls():
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from kev.api import SystemOneRequest
    from kev.serve import prepare
    q = {"q": {"type": "noul", "instructions": "i"}}
    for m in ({"type": "image", "path": "/etc/passwd"}, {"type": "image", "url": "http://169.254.169.254/"}):
        with pytest.raises(HTTPException): prepare(SystemOneRequest.model_validate({"state": "s", "questions": q, "media": [m]}))
    ok = prepare(SystemOneRequest.model_validate({"state": "s", "questions": q, "media": [{"type": "image", "data": "eA=="}]}))
    assert ok.media[0].data == "eA=="


def test_training_augmentation_keeps_media():
    """augment, none_pair and the permutation copy rebuild each record every epoch; they once kept only state and
    questions, so every record trained without its media (images still scored well zero-shot; audio stayed at chance)."""
    import random
    from kev.data import augment, materialize, none_pair
    from kev.train import permuted_copy
    media = [{"type": "audio", "path": "/x.wav"}]
    req = {"state": "", "media": media, "_meta": {"id": "a"}, "questions": {"intent": {
        "type": "choice", "instructions": "What?", "criteria": {"a": None, "b": None, "c": None, "d": None}, "label": "a", "src": "t"}}}
    rng = random.Random(0)
    assert augment(req, rng)["media"] == media and materialize(augment(req, rng))["media"] == media
    assert all(v["media"] == media for v in none_pair(req, rng))
    rec = materialize(req); rec["questions"][0]["qtype"] = "choice"
    assert permuted_copy(rec, rng)[0]["media"] == media
