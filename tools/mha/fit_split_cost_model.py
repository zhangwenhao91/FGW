"""Fit a nonnegative relative-cost model from complete split scans (offline NumPy only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from operators.mha.split_cost_model import features

ROOT = Path(__file__).resolve().parents[2]


def fit(cases):
    matrix, target = [], []
    for case in cases:
        rank = {x["num_splits"]: x["median_round_us"] for x in case["ranking"]}
        weight = 1 / min(rank.values())
        ref = np.array(features(case["B"], case["Q"], case["H"], case["kv_tiles"], 1))
        for split, latency in rank.items():
            if split != 1:
                vector = np.array(features(case["B"], case["Q"], case["H"], case["kv_tiles"], split))
                matrix.append((vector-ref)*weight)
                target.append((latency-rank[1])*weight)
    matrix, target = np.array(matrix), np.array(target)
    scale = np.linalg.norm(matrix, axis=0)
    scale[scale == 0] = 1
    matrix /= scale
    weights = np.zeros(matrix.shape[1])
    residual = target.copy()
    for iteration in range(20000):
        old = weights.copy()
        for j in range(len(weights)):
            column = matrix[:, j]
            norm = column @ column
            if norm:
                value = max(0, weights[j] + column @ residual / norm)
                residual -= column * (value-weights[j])
                weights[j] = value
        if np.max(abs(weights-old)) < 1e-9:
            break
    return (weights/scale).tolist()


def evaluate(weights, cases):
    rows = []
    for case in cases:
        rank = {x["num_splits"]: x["median_round_us"] for x in case["ranking"]}
        scores = {s: sum(w*x for w, x in zip(weights,
            features(case["B"], case["Q"], case["H"], case["kv_tiles"], s))) for s in rank}
        selected = min(scores, key=lambda s: (scores[s], s))
        best = min(rank.values())
        rows.append({"B": case["B"], "H": case["H"], "Q": case["Q"], "KV": case["KV"],
            "selected": selected, "best": case["best_num_splits"], "auto": case["auto_num_splits"],
            "relative_to_best": rank[selected]/best,
            "legacy_relative_to_best": rank[case["auto_num_splits"]]/best,
            "speedup_vs_legacy": rank[case["auto_num_splits"]]/rank[selected]})
    ratios = [r["relative_to_best"] for r in rows]
    return {"cases": len(rows), "within_2pct": sum(r <= 1.02 for r in ratios),
        "geomean_relative_to_best": math.exp(sum(math.log(r) for r in ratios)/len(ratios)),
        "max_relative_to_best": max(ratios), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/mha/latest")
    parser.add_argument("--model-output", type=Path, default=ROOT / "operators/mha/split_cost_model.json")
    parser.add_argument("--evaluation-output", type=Path, default=ROOT / "docs/mha_split_cost_model_evaluation.json")
    args = parser.parse_args()
    cases, sources = [], []
    for kv in (512, 2048, 8192, 32768):
        path = args.report_dir / f"mha_v7_split_scan_kv{kv}_summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["status"] != "passed":
            raise ValueError(f"Incomplete source scan: {path}")
        for case in data["best_by_case"]:
            if not case["all_candidates_passed"]:
                raise ValueError("Candidate accuracy failure")
            cases.append(case)
        sources.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    weights = fit(cases)
    model = {"version": 1, "scope": "experimental 28-Cube Ascend950 BF16/D128 page128 noncausal, uniform full KV",
        "features": ["compute_wave_tiles", "compute_waves", "extra_launch", "partial_elements_scaled",
                     "merge_work_times_splits", "multi_split_wave_tiles"],
        "weights": weights, "training_cases": len(cases), "sources": sources,
        "fit": "Nonnegative coordinate descent on within-case latency differences, weighted by inverse best latency. Scores omit case-constant cost and are not absolute latency."}
    evaluation = {"resubstitution": evaluate(weights, cases),
        "leave_one_kv_out": {}, "leave_one_bh_out": {},
        "caveat": "Exploratory grouped cross-validation of existing data; independent NPU validation is still required."}
    for kv in (512, 2048, 8192, 32768):
        evaluation["leave_one_kv_out"][str(kv)] = evaluate(fit([c for c in cases if c["KV"] != kv]), [c for c in cases if c["KV"] == kv])
    for batch, heads in ((1, 1), (2, 8)):
        test = [c for c in cases if (c["B"], c["H"]) == (batch, heads)]
        train = [c for c in cases if (c["B"], c["H"]) != (batch, heads)]
        evaluation["leave_one_bh_out"][f"{batch}x{heads}"] = evaluate(fit(train), test)
    for path, data in ((args.model_output, model), (args.evaluation_output, evaluation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"weights": weights, "resubstitution": {k:v for k,v in evaluation["resubstitution"].items() if k != "rows"},
        "kv_holdout": {k:{a:b for a,b in v.items() if a != "rows"} for k,v in evaluation["leave_one_kv_out"].items()},
        "bh_holdout": {k:{a:b for a,b in v.items() if a != "rows"} for k,v in evaluation["leave_one_bh_out"].items()}}, indent=2))


if __name__ == "__main__":
    main()
