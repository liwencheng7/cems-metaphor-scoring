"""Summarize Wang nested-CV predictions on the native 1–7 norm scale."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TRAITS = ("familiarity", "meaningfulness", "figurativeness")
METHODS = ("qwen", "llm_calibrated", "bge_knn", "bge_ridge", "learned_qwen_bge")
COMPARISONS = (("qwen", "learned_qwen_bge"), ("llm_calibrated", "learned_qwen_bge"))


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        output[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return output


def qwk7(gold: np.ndarray, pred: np.ndarray) -> float | None:
    if not len(gold):
        return None
    g = np.clip(np.rint(gold), 1, 7).astype(int)
    p = np.clip(np.rint(pred), 1, 7).astype(int)
    observed = np.zeros((7, 7), dtype=np.float64)
    for a, b in zip(g, p):
        observed[a - 1, b - 1] += 1
    hist_g, hist_p = np.bincount(g - 1, minlength=7), np.bincount(p - 1, minlength=7)
    expected = np.outer(hist_g, hist_p) / len(g)
    weights = ((np.arange(7)[:, None] - np.arange(7)[None, :]) / 6) ** 2
    denominator = float((weights * expected).sum())
    return None if denominator == 0 else float(1 - (weights * observed).sum() / denominator)


def trait_metrics(gold5: np.ndarray, pred5: np.ndarray) -> dict[str, float | None]:
    # Fixed inverse of the pre-registered endpoint mapping: 1 + (x_7-1)*4/6.
    gold = 1 + (gold5 - 1) * 1.5
    pred = 1 + (pred5 - 1) * 1.5
    return {"n": int(len(gold)), "mae_1_7": float(np.mean(np.abs(gold - pred))),
            "rmse_1_7": float(np.sqrt(np.mean((gold - pred) ** 2))),
            "pearson": pearson(gold, pred), "spearman": pearson(ranks(gold), ranks(pred)),
            "qwk_rounded_1_7": qwk7(gold, pred)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="nested_semantic.json from the completed semantic CV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError("Semantic nested-CV output is not complete")
    records = result.get("predictions", [])
    if len(records) != 356:
        raise ValueError(f"Expected 356 outer-test predictions, found {len(records)}")
    if any("learned_qwen_bge" not in item["scores"] for item in records):
        raise ValueError("Full Qwen+BGE predictions are incomplete")

    metrics: dict[str, dict[str, dict[str, float | None]]] = {}
    by_trait: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for method in METHODS:
        metrics[method], by_trait[method] = {}, {}
        for trait in TRAITS:
            gold5 = np.asarray([float(item["gold"][trait]) for item in records])
            pred5 = np.asarray([float(item["scores"][method][trait]) for item in records])
            metrics[method][trait] = trait_metrics(gold5, pred5)
            by_trait[method][trait] = {"gold": 1 + (gold5 - 1) * 1.5,
                                       "pred": 1 + (pred5 - 1) * 1.5}
        metrics[method]["macro"] = {
            key: (float(np.mean([metrics[method][trait][key] for trait in TRAITS
                                 if metrics[method][trait][key] is not None]))
                  if any(metrics[method][trait][key] is not None for trait in TRAITS) else None)
            for key in ("mae_1_7", "rmse_1_7", "pearson", "spearman", "qwk_rounded_1_7")
        }

    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        if not item.get("group_id"):
            raise ValueError("A prediction lacks its family group ID")
        groups[str(item["group_id"])].append(index)
    if len(groups) != 120:
        raise ValueError(f"Expected 120 family groups, found {len(groups)}")

    rng = np.random.default_rng(args.seed)
    group_ids = sorted(groups)
    differences = {f"{full}_minus_{base}": {trait: [] for trait in (*TRAITS, "macro")}
                   for base, full in COMPARISONS}
    for _ in range(args.bootstrap):
        draw = rng.integers(0, len(group_ids), size=len(group_ids))
        indices = np.asarray([index for choice in draw for index in groups[group_ids[int(choice)]]], dtype=int)
        for base_name, full_name in COMPARISONS:
            comparison = f"{full_name}_minus_{base_name}"
            trait_diffs = []
            for trait in TRAITS:
                full = by_trait[full_name][trait]
                base = by_trait[base_name][trait]
                delta = float(np.mean(np.abs(full["gold"][indices] - full["pred"][indices]))
                              - np.mean(np.abs(base["gold"][indices] - base["pred"][indices])))
                differences[comparison][trait].append(delta)
                trait_diffs.append(delta)
            differences[comparison]["macro"].append(float(np.mean(trait_diffs)))

    ci = {comparison: {trait: {
              "comparison": f"{comparison}; negative favors full method",
              "mean_delta_mae_1_7": float(np.mean(values)),
              "percentile_95_ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
              "bootstrap_replicates": args.bootstrap, "resampling_unit": "family group"}
              for trait, values in trait_values.items()}
          for comparison, trait_values in differences.items()}

    by_condition: dict[str, dict[str, Any]] = {}
    # The prediction records currently carry sentence type in their source metadata in the CSV.
    # Reconstruct the condition from stable sample IDs to keep the output independent of row order.
    for condition in ("novel_metaphor", "novel_simile", "literal"):
        subset = [item for item in records if f"_{condition}" in item["sample_id"]]
        if subset:
            by_condition[condition] = {method: {
                trait: trait_metrics(
                    np.asarray([float(item["gold"][trait]) for item in subset]),
                    np.asarray([float(item["scores"][method][trait]) for item in subset]))
                for trait in TRAITS} for method in METHODS}

    output = {"status": "completed", "source": str(args.input), "run": result.get("run"),
              "metrics_native_scale_1_7": metrics, "paired_family_bootstrap": ci,
              "condition_metrics_native_scale_1_7": by_condition,
              "bootstrap_seed": args.seed,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": output["status"], "n": len(records), "groups": len(groups),
                      "macro_metrics": {method: metrics[method]["macro"] for method in METHODS},
                      "paired_family_bootstrap": ci}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
