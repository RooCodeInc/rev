"""Zero-shot baseline for records with media: the untrained base answers each question as a lettered multiple choice,
with the record's images/audio/video in the prompt, read from its next-token logits over the letters (the multimodal
counterpart of scripts/base_mmlu_probe.py's plain prompt). Writes rows in kev.benchmark's shape (task, label, p) so
scripts/mm_report.py can put it next to trained and blind runs.

Run: uv run --extra media python scripts/base_mm_probe.py --base google/gemma-4-E4B --revision <sha> --data data/mm-v0/test.jsonl --out runs/probes/e4b-mm-v0
"""
import argparse, json
from pathlib import Path

import torch

from kev.data import load_records, materialize
from kev.media import load
from kev.suite import write_json

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--revision", default=None); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="mps"); ap.add_argument("--max_options", type=int, default=26)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.base, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.base, revision=a.revision, dtype=torch.bfloat16).to(a.device).eval()
    tok = proc.tokenizer
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    rows, skipped = [], 0
    for r in load_records(a.data):
        rec = materialize(r); items = [load(m) for m in rec.get("media") or []]
        placeholders = "".join({"image": proc.image_token, "audio": proc.audio_token, "video": proc.video_token}[m["type"]] for m in items)
        kw = {"images": [m["data"] for m in items if m["type"] == "image"] or None, "audio": [m["data"] for m in items if m["type"] == "audio"] or None,
              "videos": [m["data"] for m in items if m["type"] == "video"] or None}
        for q in rec["questions"]:
            K = len(q["options"])
            if K > a.max_options: skipped += 1; continue
            opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
            prompt = f"{placeholders}\n{rec['state']}\nQuestion: {q['instr']}\nOptions:\n{opts}\nAnswer:"
            inputs = proc(text=prompt, **{k: v for k, v in kw.items() if v}, return_tensors="pt").to(a.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            p = torch.softmax(logits[letter_ids[:K]], -1).tolist()
            rows.append({"id": r["_meta"]["id"], "task": q["src"], "source": r["_meta"]["source"], "label": q["label"], "p": p, "type": q["qtype"]})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    acc = sum(max(range(len(x["p"])), key=x["p"].__getitem__) == x["label"] for x in rows) / max(len(rows), 1)
    report = {"base": a.base, "revision": a.revision, "data": a.data, "readout": "zero-shot next-token letter logits",
              "questions": len(rows), "skipped_over_max_options": skipped, "accuracy": acc}
    write_json(out / "report.json", report)
    print(f"{len(rows)} questions ({skipped} skipped: more than {a.max_options} options), accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
