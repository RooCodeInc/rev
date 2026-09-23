"""Ask a running server (kev.serve) one question about a file: an image, an audio clip or a video, sent inline.

    uv run python scripts/ask.py photo.jpg "Is there a dog in the photo?"
    uv run python scripts/ask.py call.wav "What does the caller want?" --options balance,card_issues,freeze,other
    uv run python scripts/ask.py clip.mp4 "How urgent is this?" --levels "not urgent,soon,right now"

No --options/--levels: a yes/no question. Start the server first:
    uv run --extra serve --extra media python -m kev.serve --run runs/rev-e4b-mm-v0 --port 8009
"""
import argparse, base64, json, mimetypes, urllib.request
from pathlib import Path

KINDS = {"image": "image", "audio": "audio", "video": "video"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("question")
    ap.add_argument("--options", help="comma-separated choices (a choice question)")
    ap.add_argument("--levels", help="comma-separated ordered levels, lowest first (a score question)")
    ap.add_argument("--state", default="", help="text that goes with the file")
    ap.add_argument("--url", default="http://127.0.0.1:8009/v1/systemone")
    a = ap.parse_args()
    mime = mimetypes.guess_type(a.file)[0] or ""
    kind = KINDS.get(mime.split("/")[0])
    if kind is None: ap.error(f"cannot tell whether {a.file} is an image, audio or video ({mime or 'unknown type'})")
    if a.options: q = {"type": "choice", "instructions": a.question, "criteria": {o.strip(): None for o in a.options.split(",")}}
    elif a.levels: q = {"type": "score", "instructions": a.question, "criteria": [x.strip() for x in a.levels.split(",")]}
    else: q = {"type": "noul", "instructions": a.question}
    body = {"state": a.state, "questions": {"q": q}, "media": [{"type": kind, "data": base64.b64encode(Path(a.file).read_bytes()).decode()}]}
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    print(json.dumps(out["answers"]["q"], indent=1))
    print(f"({out['latency_ms']} ms, {out['usage']['input_tokens']} input tokens)")


if __name__ == "__main__":
    main()
