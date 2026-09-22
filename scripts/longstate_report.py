"""Accuracy by state length on evals/round4/longstate-v1 (PLAN.md round 4, item 4.12; PLAN_27b A3), paired against the
same primaries unburied (longstate_control), from a kev.benchmark rows.json.

    uv run python scripts/longstate_report.py runs/r4-kev-4b-longstate-2 [runs/<other> ...]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import read_json, write_json  # noqa: E402


def by_length(rows):
    """{length: (accuracy buried, accuracy of the same questions unburied, n, paired 95% bootstrap CI of the drop)}"""
    ok = lambda r: int(np.argmax(r["p"]) == r["label"])
    control = {(r["group"], r["question"]): ok(r) for r in rows if r["source"] == "longstate_control" and r["variant"] == "clean"}
    pairs = defaultdict(list)
    for r in rows:
        if r["source"] == "longstate" and r["variant"] == "clean" and (r["group"], r["question"]) in control:
            pairs[r["task"].split("_")[1]].append((ok(r), control[(r["group"], r["question"])]))
    rng, out = np.random.default_rng(0), {}
    for length, p in sorted(pairs.items(), key=lambda kv: int(kv[0])):
        a = np.asarray(p, dtype=float); d = a[:, 0] - a[:, 1]
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out[length] = {"n": len(a), "buried": float(a[:, 0].mean()), "unburied": float(a[:, 1].mean()), "delta": float(d.mean()),
                       "ci95": np.quantile(boots, [0.025, 0.975]).tolist()}
    return out


def main():
    for run in sys.argv[1:]:
        report = by_length(read_json(Path(run) / "rows.json"))
        write_json(Path(run) / "longstate.json", report)
        print(run)
        for length, r in report.items():
            print(f"  {length:>5} tokens  n={r['n']:3}  buried {r['buried']:.3f}  unburied {r['unburied']:.3f}  delta {r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
