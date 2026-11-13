"""Fit/reproduce the general Cube model from a compact same-process split scan.

NumPy only. Leave-one-shape-out removes both dtypes and all splits together.
The measured single/multi call floors are recomputed inside each training fold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from operators.mha.general_split_cost_model import feature_names, features, select_model_splits


def fit(rows, version=1):
    import numpy as np
    x = np.array([features(r["B"], r["Q"], r["H"], r["D"], r["lengths"], int(s), version)
                  for r in rows for s in r["median_us"]], dtype=float)
    y = np.array([us for r in rows for us in r["median_us"].values()], dtype=float)
    # Minimize squared relative latency error with nonnegative coefficients.
    a = x/y[:, None]
    target = np.ones(len(y))
    if version == 2:
        # Within-shape contrasts target split ranking and reduce sensitivity to
        # measurement-session constant offsets. Fixed weight, not a case lookup.
        contrast_x, contrast_y = [], []
        for r in rows:
            values = r["median_us"]
            labels = list(values)
            if len(labels) < 2:
                continue
            vectors = {s: np.array(features(r["B"], r["Q"], r["H"], r["D"], r["lengths"], int(s), version)) for s in labels}
            factor = math.sqrt(0.25/(len(labels)-1))/min(values.values())
            for i, s in enumerate(labels):
                for t in labels[i+1:]:
                    contrast_x.append((vectors[s]-vectors[t])*factor)
                    contrast_y.append((values[s]-values[t])*factor)
        a = np.vstack([a, np.array(contrast_x)])
        target = np.concatenate([target, np.array(contrast_y)])
    scale = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    a /= scale
    weights = np.zeros(x.shape[1])
    residual = target.copy()
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
    return {"version": version, "feature_names": list(feature_names(version)), "weights": (weights/scale).tolist(),
            "latency_floor_us": {
                mode: min(us for r in rows for s, us in r["median_us"].items()
                          if (int(s) > 1) == multi)
                for mode, multi in (("single", False), ("multi", True))},
            "selection_tolerance": 0.02, "absolute_selection_margin_us": 1.0 if version == 2 else 0.0,
            "contrast_loss_weight": 0.25 if version == 2 else 0.0,
            "coordinate_descent_iterations": iterations+1}


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
    parser.add_argument("--version", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    source = Path(args.input)
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["rows"]
    model = fit(rows, args.version)
    fitted = evaluate(rows, model)
    folds = []
    for shape in sorted({r["case"] for r in rows}):
        train = [r for r in rows if r["case"] != shape]
        held = [r for r in rows if r["case"] == shape]
        folds.extend(evaluate(held, fit(train, args.version)))
    hash_bytes = source.read_text(encoding="utf-8").encode("utf-8") if args.version == 2 else source.read_bytes()
    model.update(status="experimental_not_default", hardware="Ascend950PR_9579 / 28 Cube cores",
                 training_sha256=hashlib.sha256(hash_bytes).hexdigest(),
                 training_hash_basis="UTF-8/LF text" if args.version == 2 else "Raw fitter working-copy bytes",
                 training_rows=len(rows), training_shapes=len({r["case"] for r in rows}),
                 training_variants=sum(len(r["median_us"]) for r in rows),
                 source_precision_us=data["source_precision_us"],
                 fitting="Relative-error least squares; optional within-shape split contrasts; nonnegative coordinate descent; empirical single/multi latency floors",
                 validation={"calibration": summarize(fitted), "leave_one_shape_out": summarize(folds)},
                 limitations="Rounded pasted medians, not raw samples. LOSO is offline validation; independent NPU holdout pending. No default promotion.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"weights": model["weights"], "floors": model["latency_floor_us"],
                      "calibration": {k: v for k, v in model["validation"]["calibration"].items() if k != "results"},
                      "leave_one_shape_out": {k: v for k, v in model["validation"]["leave_one_shape_out"].items() if k != "results"}}, indent=2))


if __name__ == "__main__":
    main()
