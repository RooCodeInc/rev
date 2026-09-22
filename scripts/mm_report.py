"""Per-task accuracy for a multimodal test set, side by side: the trained model, the same model with the media removed
("blind": what the text alone gives it), the untrained base answering zero-shot (scripts/base_mm_probe.py), and chance.
A task the model solves from its media shows trained >> blind ~ chance.

    uv run python scripts/mm_report.py --runs trained=runs/x-bench blind=runs/x-blind base=runs/probes/e4b-mm-v0 [--out report.md]

Also: --blind_data <test.jsonl> writes <test>-blind.jsonl (the same records without media) and exits.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


def per_task(rows_path):
    acc, chance = defaultdict(list), defaultdict(list)
    for r in json.loads(Path(rows_path).read_text(encoding="utf-8")):
        p = r["p"]; acc[r["task"]].append(max(range(len(p)), key=p.__getitem__) == r["label"]); chance[r["task"]].append(1 / len(p))
    return {t: (sum(v) / len(v), len(v)) for t, v in acc.items()}, {t: sum(v) / len(v) for t, v in chance.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="name=dir pairs; each dir has rows.json")
    ap.add_argument("--out"); ap.add_argument("--blind_data")
    a = ap.parse_args()
    if a.blind_data:
        src = Path(a.blind_data); dst = src.with_name(src.stem + "-blind.jsonl")
        with open(dst, "w", encoding="utf-8") as f:
            for line in src.read_text(encoding="utf-8").splitlines():
                r = json.loads(line); r.pop("media", None); f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {dst}"); return
    cols = [(n, *per_task(Path(d) / "rows.json")) for n, d in (x.split("=", 1) for x in a.runs)]
    tasks = sorted({t for _, acc, _ in cols for t in acc})
    chance = {t: next(c[t] for _, acc, c in cols if t in c) for t in tasks}
    lines = ["| task | n | " + " | ".join(n for n, _, _ in cols) + " | chance |", "|---|---|" + "---|" * (len(cols) + 1)]
    for t in tasks:
        n = next(acc[t][1] for _, acc, _ in cols if t in acc)
        lines.append(f"| {t} | {n} | " + " | ".join(f"{acc[t][0]:.3f}" if t in acc else "–" for _, acc, _ in cols) + f" | {chance[t]:.3f} |")
    macro = [sum(acc[t][0] for t in tasks if t in acc) / max(1, sum(t in acc for t in tasks)) for _, acc, _ in cols]
    lines.append("| **macro** | | " + " | ".join(f"**{m:.3f}**" for m in macro) + f" | {sum(chance.values()) / len(chance):.3f} |")
    text = "\n".join(lines); print(text)
    if a.out: Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
