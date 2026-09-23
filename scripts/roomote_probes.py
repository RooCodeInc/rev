"""Roomote's skill-relevance probes (Roomote-Cloud scripts/judgment-data/probe.py) run in-process on a checkpoint.

Each probe sends Roomote's real rerank shape: one shared state {query, candidates: {k0..k7}} and one yes/no question
per skill naming it by reference (`candidates.k5`). A probe passes when the relevant skill scores >= 0.7 and >= 0.25
above the runner-up (Fast's suggestion rule). The live Kev model cannot follow the reference, which is why its server
splits reranks into one candidate per row; this measures the shared form directly, and the split form as a second
column (what the serving script sends Kev).

    uv run --extra serve --extra media python scripts/roomote_probes.py --run runs/rev-e4b-text-v7 --probes data/roomote/judgment-data/probes.json
"""
import argparse, json
from pathlib import Path

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.device import default_device
from kev.model import training_context

SUGGESTED_MIN, SUGGESTED_MARGIN = 0.7, 0.25


def yes(model, tok, state, questions, context):
    rec, meta = to_record(SystemOneRequest.model_validate({"state": state, "questions": questions}))
    enc = model.encode(tok, rec, max_state=context["max_state"], max_branch=context["max_branch"])
    return {m["id"]: float(p[1]) for p, m in zip(model.probs(enc), meta)}   # noul options are [no, yes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--probes", nargs="+", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    dev = default_device(); tok, model = Checkpoint(a.run).load(dev, LoadOptions.from_env())
    context = training_context(2048); report = {"run": a.run}
    for path in a.probes:
        spec = json.loads(Path(path).read_text(encoding="utf-8")); name = Path(path).stem
        shared_pass = split_pass = 0; lines = []
        for probe in spec["probes"]:
            questions = {k: {"type": "noul", "instructions": spec["instructions"].format(key=k)} for k in spec["candidates"]}
            shared = yes(model, tok, {"query": probe["query"], "candidates": spec["candidates"]}, questions, context)
            split = {k: yes(model, tok, {"query": probe["query"], "candidates": {k: v}}, {k: questions[k]}, context)[k] for k, v in spec["candidates"].items()}
            row = {}
            for form, ans in (("shared", shared), ("split", split)):
                right = ans[probe["relevant"]]; wrong = max(v for k, v in ans.items() if k != probe["relevant"])
                ok = right >= SUGGESTED_MIN and right - wrong >= SUGGESTED_MARGIN
                row[form] = {"relevant": round(right, 3), "best_other": round(wrong, 3), "pass": ok}
            shared_pass += row["shared"]["pass"]; split_pass += row["split"]["pass"]; lines.append({"query": probe["query"], **row})
            print(f"{name}  shared {'pass' if row['shared']['pass'] else 'FAIL'} {row['shared']['relevant']:.2f}/{row['shared']['best_other']:.2f}"
                  f"  split {'pass' if row['split']['pass'] else 'FAIL'} {row['split']['relevant']:.2f}/{row['split']['best_other']:.2f}  {probe['query'][:55]}", flush=True)
        n = len(spec["probes"]); print(f"{name}: shared {shared_pass}/{n}, split {split_pass}/{n}")
        report[name] = {"shared": shared_pass, "split": split_pass, "n": n, "probes": lines}
    if a.out: Path(a.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
