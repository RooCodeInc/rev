"""Accuracy on the Roomote judgment holdout, overall and by slice: captured (real traffic) vs synthetic, and per
decision surface, for several scored runs side by side. Rows come from kev.benchmark --data <holdout.jsonl>; their id
is `custom/<line>`, which maps back to the holdout row's _meta (surface, origin).

    uv run python scripts/roomote_report.py --holdout data/roomote/holdout.jsonl --runs kev_roomote=runs/roomote-holdout-kev-roomote-v2 ... [--out x.md]
The holdout is private: keep its files and per-row outputs out of git (data/ and rows.json are ignored).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--holdout", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    meta = [json.loads(l)["_meta"] for l in Path(a.holdout).read_text(encoding="utf-8").splitlines() if l.strip()]
    cols = []
    for spec in a.runs:
        name, d = spec.split("=", 1); p = Path(d) / "rows.json"
        if not p.exists(): continue
        hits = defaultdict(list)
        for r in json.loads(p.read_text(encoding="utf-8")):
            m = meta[int(r["id"].rsplit("/", 1)[1])]; p_ = r["p"]; ok = max(range(len(p_)), key=p_.__getitem__) == r["label"]
            for key in ("all", m["origin"], "surface:" + m["surface"]): hits[key].append(ok)
        rep = json.loads((Path(d) / "report.json").read_text(encoding="utf-8"))["coverage"]
        cols.append((name, {k: sum(v) / len(v) for k, v in hits.items()}, {k: len(v) for k, v in hits.items()}, rep))
    keys = ["all", "captured", "synthetic"] + sorted({k for _, acc, _, _ in cols for k in acc if k.startswith("surface:")})
    lines = ["| Slice | n | " + " | ".join(n for n, *_ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k in keys:
        n = max(c[2].get(k, 0) for c in cols)
        vals = [c[1].get(k) for c in cols]; best = max((v for v in vals if v is not None), default=None)
        label = {"all": "**All**", "captured": "**Captured (real traffic)**", "synthetic": "Synthetic"}.get(k, k.removeprefix("surface:"))
        lines.append(f"| {label} | {n} | " + " | ".join("–" if v is None else (f"**{v:.3f}**" if v == best else f"{v:.3f}") for v in vals) + " |")
    lines.append("| rejected (over 2,048-token states) | | " + " | ".join(str(c[3]["rejected_records"]) for c in cols) + " |")
    text = "\n".join(lines); print(text)
    if a.out: Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
