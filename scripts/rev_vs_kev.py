"""Rev vs Kev vs Jev on Kev's text benchmarks (development partitions): clean accuracy per suite, from saved reports.
Kev and Jev columns are the committed reference runs; Kev's transfer-v4 / decision-v7 numbers are the README's (the
release checkpoints, after the night-2 delta), the other Kev-9B cells are its v7 checkpoint (q35-9b), the same recipe
Rev's text runs use.

    uv run python scripts/rev_vs_kev.py --rev rev_text=runs/rev-e4b-text-v7 rev_combined=runs/rev-e4b-combined-v1 [--out report.md]
A Rev column reads <dir>-<suite>/report.json for each suite.
"""
import argparse, json
from pathlib import Path

SUITES = ["transfer-v4", "decision-v7", "transfer-v9", "semif-v1", "wanli-v1", "typesafe-v1", "scienthoon-v1"]
LABEL = {"transfer-v4": "New sources (transfer-v4)", "decision-v7": "Trained sources (decision-v7)", "transfer-v9": "transfer-v9 (MMLU-Pro, buried, unknowable)",
         "semif-v1": "SemIf authored decisions", "wanli-v1": "WANLI", "typesafe-v1": "TypeSafe public questions", "scienthoon-v1": "Scienthoon support tickets"}
REFS = {"Kev-4B": {"transfer-v4": 0.797, "decision-v7": 0.872, "semif-v1": "runs/kev-4b-semif-v1", "wanli-v1": "runs/kev-4b-wanli-v1",
                   "typesafe-v1": "runs/kev-4b-typesafe-v1", "scienthoon-v1": "runs/kev-4b-scienthoon-v1-gpu"},
        "Kev-9B": {"transfer-v4": 0.822, "decision-v7": 0.872, "transfer-v9": "runs/q35-9b-s0-transfer-v9", "semif-v1": "runs/q35-9b-s0-semif-v1",
                   "wanli-v1": "runs/kev-9b-wanli-v1", "typesafe-v1": "runs/kev-9b-typesafe-v1", "scienthoon-v1": "runs/q35-9b-s0-scienthoon-v1"},
        "Jev": {"transfer-v4": 0.857, "decision-v7": 0.845, "transfer-v9": "runs/jev-transfer-v9", "semif-v1": "runs/jev-semif-v1", "wanli-v1": "runs/jev-wanli-v1",
                "typesafe-v1": "runs/jev-typesafe-v1", "scienthoon-v1": "runs/jev-scienthoon-v1"}}


def acc(cell):
    if isinstance(cell, (int, float)): return cell
    p = Path(cell) / "report.json"
    return json.loads(p.read_text(encoding="utf-8"))["clean"]["acc"] if p.exists() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rev", nargs="+", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    cols = [(n, {s: f"{d}-{s}" for s in SUITES}) for n, d in (x.split("=", 1) for x in a.rev)] + list(REFS.items())
    lines = ["| Benchmark | " + " | ".join(n for n, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for s in SUITES:
        vals = [acc(c[s]) if s in c else None for _, c in cols]
        best = max((v for v in vals if v is not None), default=None)
        cells = [("–" if v is None else (f"**{v:.3f}**" if v == best else f"{v:.3f}")) for v in vals]
        lines.append(f"| {LABEL[s]} | " + " | ".join(cells) + " |")
    text = "\n".join(lines); print(text)
    if a.out: Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
