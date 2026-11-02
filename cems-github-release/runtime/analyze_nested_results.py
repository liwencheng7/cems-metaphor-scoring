"""Audit nested-CV outputs and compute paired group-bootstrap MAE intervals."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import scorer

DIMENSIONS = tuple(scorer.DIMENSIONS)
PREDECLARED_BASELINES = ("qwen", "char_tfidf_knn")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def quantile_ci(values: list[float]) -> dict[str, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return {"lower_95": float(low), "upper_95": float(high)}


def validate_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("status") != "completed":
        raise ValueError(f"Nested result is not completed: {result.get('status')}")
    records = result.get("predictions")
    if not isinstance(records, list) or not records:
        raise ValueError("No outer-fold predictions found")
    ids = [row.get("sample_id") for row in records]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("Outer predictions have missing or duplicated sample IDs")
    for row in records:
        if not row.get("group_id"):
            raise ValueError(f"Prediction lacks group_id: {row.get('sample_id')}")
        for dim in DIMENSIONS:
            if dim not in row.get("gold", {}) or not np.isfinite(float(row["gold"][dim])):
                raise ValueError(f"Invalid gold value for {row['sample_id']} / {dim}")
            for method, values in row.get("scores", {}).items():
                if dim not in values or not np.isfinite(float(values[dim])):
                    raise ValueError(f"Invalid prediction for {row['sample_id']} / {method} / {dim}")
    methods = set(records[0]["scores"])
    if any(set(row["scores"]) != methods for row in records):
        raise ValueError("Prediction methods are inconsistent across records")
    if not set(PREDECLARED_BASELINES).issubset(methods):
        raise ValueError(f"Required direct baselines are missing: {PREDECLARED_BASELINES}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True,
                        help="Completed nested_cv.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates; the frozen plan uses 2,000")

    source = args.input_json.resolve()
    result = read_json(source)
    records = validate_predictions(result)
    methods = sorted(records[0]["scores"])
    n = len(records)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(str(row["group_id"]), []).append(index)
    group_ids = sorted(groups)
    dim_index = {dim: j for j, dim in enumerate(DIMENSIONS)}
    gold = np.asarray([[float(row["gold"][dim]) for dim in DIMENSIONS] for row in records], dtype=np.float64)
    predictions = {method: np.asarray([[float(row["scores"][method][dim]) for dim in DIMENSIONS]
                                       for row in records], dtype=np.float64)
                   for method in methods}

    point = {}
    for baseline in PREDECLARED_BASELINES:
        if baseline not in methods:
            continue
        point[baseline] = {}
        for method in methods:
            if method == baseline:
                continue
            per_dimension = {}
            for dim, j in dim_index.items():
                delta = float(np.mean(np.abs(predictions[method][:, j] - gold[:, j]) -
                                      np.abs(predictions[baseline][:, j] - gold[:, j])))
                per_dimension[dim] = delta
            point[baseline][method] = {"per_dimension_delta_mae": per_dimension,
                                       "macro_delta_mae": float(np.mean(list(per_dimension.values())))}

    rng = np.random.default_rng(args.seed)
    draws: dict[str, dict[str, dict[str, list[float]]]] = {
        baseline: {method: {"macro": [], **{dim: [] for dim in DIMENSIONS}}
                   for method in methods if method != baseline}
        for baseline in PREDECLARED_BASELINES if baseline in methods
    }
    group_array = np.asarray(group_ids, dtype=object)
    for _ in range(args.bootstrap_replicates):
        sampled_groups = rng.choice(group_array, size=len(group_array), replace=True)
        indices = np.fromiter((idx for group_id in sampled_groups for idx in groups[str(group_id)]),
                              dtype=np.int64)
        losses = {method: np.abs(predictions[method][indices] - gold[indices]) for method in methods}
        for baseline, method_draws in draws.items():
            base_mae = losses[baseline].mean(axis=0)
            for method in method_draws:
                delta_by_dim = losses[method].mean(axis=0) - base_mae
                for j, dim in enumerate(DIMENSIONS):
                    method_draws[method][dim].append(float(delta_by_dim[j]))
                method_draws[method]["macro"].append(float(delta_by_dim.mean()))

    intervals = {}
    for baseline, method_draws in draws.items():
        intervals[baseline] = {}
        for method, metric_draws in method_draws.items():
            intervals[baseline][method] = {}
            for metric_name, values in metric_draws.items():
                intervals[baseline][method][metric_name] = {
                    **quantile_ci(values),
                    "probability_delta_mae_below_zero": float(np.mean(np.asarray(values) < 0.0)),
                }

    output = {"status": "completed", "source": str(source),
              "protocol": "paired_group_bootstrap_oof_mae_delta",
              "n": n, "n_groups": len(group_ids),
              "group_size_distribution": {"min": min(map(len, groups.values())),
                                          "median": float(np.median(list(map(len, groups.values())))),
                                          "max": max(map(len, groups.values()))},
              "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
              "baselines": list(PREDECLARED_BASELINES),
              "delta_definition": "method MAE minus baseline MAE; negative favors method",
              "point_estimates": point, "paired_95_percentile_intervals": intervals,
              "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(args.output_json.resolve(), output)
    print(json.dumps({"status": output["status"], "n": n, "n_groups": len(group_ids),
                      "bootstrap_replicates": args.bootstrap_replicates,
                      "output": str(args.output_json.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
