"""Fit/reproduce the general Cube model from a compact same-process split scan.

NumPy only. Leave-one-shape-out removes both dtypes and all splits together.
The measured single/multi call floors are recomputed inside each training fold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from operators.mha.general_split_cost_model import FEATURE_NAMES, features, select_model_splits


def fit(rows):
    import numpy as np
    x = np.array([features(r["B"], r["Q"], r["H"], r["D"], r["lengths"], int(s))
                  for r in rows for s in r["median_us"]], dtype=float)
    y = np.array([us for r in rows for us in r["median_us"].values()], dtype=float)
    # Minimize squared relative latency error with nonnegative coefficients.
    a = x/y[:, None]
    scale = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    a /= scale
    weights = np.zeros(x.shape[1])
    residual = np.ones(len(y))
    norms = np.sum(a*a, axis=0)
    iterations = 0
    for iterations in range(8000):
        old = weights.copy()
        for j in range(len(weights)):
            update = max(0, weights[j] + a[:, j] @ residual / max(norms[j], 1e-20))
            residual += a[:, j]*(weights[j]-update)
            weights[j] = update
        if np.max(np.abs(weights-old)) < 1e-9:
            break
    return {"feature_names": list(FEATURE_NAMES), "weights": (weights/scale).tolist(),
            "latency_floor_us": {
                mode: min(us for r in rows for s, us in r["median_us"].items()
                          if (int(s) > 1) == multi)
                for mode, multi in (("single", False), ("multi", True))},
            "selection_tolerance": 0.02, "coordinate_descent_iterations": iterations+1}


def evaluate(rows, model):
    results = []
    for r in rows:
        split = select_model_splits(r["B"], r["Q"], r["H"], r["D"], r["page_size"],
                                    r["table_width"], r["lengths"], model)
        selected = r["median_us"][str(split)]
        best = min(r["median_us"].values())
        results.append({"key": r["key"], "model_splits": split,
                        "regret_percent": (selected/best-1)*100})
    return results


def summarize(results):
    regrets = [r["regret_percent"] for r in results]
    return {"cases": len(results), "within_2pct": sum(v <= 2 for v in regrets),
            "max_regret_percent": max(regrets),
            "geomean_regret_percent": (math.exp(sum(math.log1p(v/100) for v in regrets)/len(regrets))-1)*100,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["rows"]
    model = fit(rows)
    fitted = evaluate(rows, model)
    folds = []
    for shape in sorted({r["case"] for r in rows}):
        train = [r for r in rows if r["case"] != shape]
        held = [r for r in rows if r["case"] == shape]
        folds.extend(evaluate(held, fit(train)))
    model.update(version=1, status="experimental_not_default", hardware="Ascend950PR_9579 / 28 Cube cores",
                 training_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 training_rows=len(rows), training_shapes=len({r["case"] for r in rows}),
                 training_variants=sum(len(r["median_us"]) for r in rows),
                 source_precision_us=data["source_precision_us"],
                 fitting="Relative-error least squares; nonnegative coordinate descent; empirical single/multi latency floors",
                 validation={"calibration": summarize(fitted), "leave_one_shape_out": summarize(folds)},
                 limitations="Rounded pasted medians, not raw samples. LOSO is offline validation; independent NPU holdout pending. No default promotion.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"weights": model["weights"], "floors": model["latency_floor_us"],
                      "calibration": {k: v for k, v in model["validation"]["calibration"].items() if k != "results"},
                      "leave_one_shape_out": {k: v for k, v in model["validation"]["leave_one_shape_out"].items() if k != "results"}}, indent=2))


if __name__ == "__main__":
    main()
