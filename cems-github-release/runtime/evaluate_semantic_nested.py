"""Evaluate a frozen Chinese encoder inside the existing leakage-controlled CV.

The Qwen/character OOF predictions are read from the completed nested run's
per-role checkpoints. Semantic vectors are text-only and are never fit on gold
labels. Each outer-fold stacker uses only the four inner OOF folds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "cogmed_items.csv"
DEFAULT_SPLITS = HERE / "data" / "splits.json"
DEFAULT_MAIN_RUN = HERE / "results" / "paper_complete_20261018" / "nested_main"
DEFAULT_OUTPUT = HERE / "results" / "paper_complete_20261018" / "semantic_bge"
DIMENSIONS = ("familiarity", "meaningfulness", "figurativeness", "valence",
              "difficulty_raw", "imageability", "naturalness")
MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
RIDGE_ALPHA = 10.0
KNN_K = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def encode_texts(texts: list[str], batch_size: int, device: str,
                 local_model_path: Path | None = None) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("E6 requires torch and transformers in the experiment environment") from error

    source = str(local_model_path) if local_model_path else MODEL_ID
    load_options = ({"local_files_only": True} if local_model_path else {"revision": MODEL_REVISION})
    tokenizer = AutoTokenizer.from_pretrained(source, **load_options)
    model = AutoModel.from_pretrained(source, use_safetensors=True, **load_options)
    model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch).last_hidden_state[:, 0]
            output = functional.normalize(output, p=2, dim=1)
            chunks.append(output.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def get_embeddings(rows: list[dict[str, str]], dataset_hash: str, cache_dir: Path,
                   batch_size: int, device: str,
                   local_model_path: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vector_path = cache_dir / "bge_small_zh_v1_5_cls_normalized.npy"
    metadata_path = cache_dir / "embedding_metadata.json"
    expected = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "dataset_sha256": dataset_hash, "pooling": "CLS",
                "normalization": "L2", "max_length": 512,
                "sample_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()}
    if vector_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if all(metadata.get(key) == value for key, value in expected.items()):
            vectors = np.load(vector_path, allow_pickle=False)
            if vectors.ndim == 2 and vectors.shape[0] == len(rows) and np.isfinite(vectors).all():
                return vectors, metadata
        raise ValueError("Embedding cache exists but its identity or contents do not match; preserve it and choose a new cache directory")
    vectors = encode_texts([row["text"] for row in rows], batch_size, device, local_model_path)
    if vectors.shape[0] != len(rows) or vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid embedding matrix shape/content: {vectors.shape}")
    np.save(vector_path, vectors, allow_pickle=False)
    metadata = {**expected, "n": len(rows), "dimension": int(vectors.shape[1]),
                "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(metadata_path, metadata)
    return vectors, metadata


def semantic_predictions(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]],
                         embeddings: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, float]]]:
    ids_train = [row["id"] for row in train_rows]
    ids_test = [row["id"] for row in test_rows]
    x_train = np.stack([embeddings[sid] for sid in ids_train]).astype(np.float64)
    x_test = np.stack([embeddings[sid] for sid in ids_test]).astype(np.float64)
    y_train = np.asarray([[float(row[dim]) for dim in DIMENSIONS] for row in train_rows], dtype=np.float64)
    mean = y_train.mean(axis=0)

    # Dual ridge with an intercept; alpha is fixed to the character-Ridge value.
    x_mean = x_train.mean(axis=0)
    centered = x_train - x_mean
    kernel = centered @ centered.T + RIDGE_ALPHA * np.eye(len(train_rows))
    dual = np.linalg.solve(kernel, y_train - mean)
    ridge = np.clip((x_test - x_mean) @ centered.T @ dual + mean, 1.0, 5.0)

    similarities = x_test @ x_train.T
    nearest = np.argsort(-similarities, axis=1, kind="stable")[:, :min(KNN_K, len(train_rows))]
    weights = np.maximum(np.take_along_axis(similarities, nearest, axis=1), 0.01)
    numerator = (y_train[nearest] * weights[:, :, None]).sum(axis=1)
    denominator = weights.sum(axis=1, keepdims=True)
    knn = np.clip(numerator / denominator, 1.0, 5.0)
    return {sid: {"bge_knn": {dim: float(knn[i, j]) for j, dim in enumerate(DIMENSIONS)},
                  "bge_ridge": {dim: float(ridge[i, j]) for j, dim in enumerate(DIMENSIONS)}}
            for i, sid in enumerate(ids_test)}


def load_role(main_run: Path, outer: int, kind: str, sample_id: str,
              inner: int | None, expected_config_hash: str) -> dict[str, Any]:
    role_dir = f"inner_{inner}" if kind == "inner_oof" else "outer_test"
    path = main_run / "items" / f"outer_{outer}" / role_dir / f"{sample_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing main-run checkpoint: {path}")
    document = read_json(path)
    role = document.get("role", {})
    if (document.get("config_hash") != expected_config_hash or
            document.get("sample_id") != sample_id or role.get("outer_fold") != outer or
            role.get("kind") != kind or role.get("inner_fold") != inner):
        raise ValueError(f"Main-run checkpoint identity mismatch: {path}")
    return document


def main() -> None:
    global DIMENSIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--main-run", type=Path, default=DEFAULT_MAIN_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scorer", type=Path, default=None,
                        help="Optional scorer adapter; its dimensions define the traits evaluated")
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--encoder-path", type=Path, default=None,
                        help="Use a previously downloaded, revision-pinned local BGE snapshot")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--primary-only", action="store_true",
                        help="Evaluate the full Qwen+BGE method and base predictors without component ablations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    dataset_path, splits_path = args.dataset.resolve(), args.splits.resolve()
    main_run, output_dir = args.main_run.resolve(), args.output_dir.resolve()
    rows, manifest = read_rows(dataset_path), read_json(splits_path)
    assignment = {str(key): int(value) for key, value in manifest["assignment"].items()}
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(assignment):
        raise ValueError("Dataset and split IDs do not match")
    if any(int(row["fold"]) != assignment[row["id"]] for row in rows):
        raise ValueError("Dataset fold values differ from split manifest")
    import run_nested_cv as nested
    scoring_module = (nested.load_scorer(args.scorer.resolve()) if args.scorer else None)
    dimensions = tuple(scoring_module.DIMENSIONS) if scoring_module else DIMENSIONS
    if not dimensions:
        raise ValueError("Scorer adapter defines no dimensions")
    DIMENSIONS = dimensions
    dataset_hash, split_hash = sha256_file(dataset_path), sha256_file(splits_path)
    outer_folds = sorted(set(assignment.values()))
    expected_roles = sum(sum(assignment[row["id"]] != outer for row in rows) for outer in outer_folds)
    if args.dry_run:
        print(json.dumps({"status": "preflight_ok", "n": len(rows), "groups": len(set(row.get("group_id") for row in rows)),
                          "traits": list(DIMENSIONS), "outer_folds": outer_folds,
                          "expected_inner_role_checkpoints": expected_roles,
                          "encoder": MODEL_ID, "revision": MODEL_REVISION,
                          "methods": (["qwen", "llm_calibrated", "bge_knn", "bge_ridge", "learned_qwen_bge"]
                                      if args.primary_only else None)}, ensure_ascii=False))
        return

    main_progress_path = main_run / "progress.json"
    if not main_progress_path.exists():
        raise FileNotFoundError(f"Main nested-run progress is missing: {main_progress_path}")
    main_progress = read_json(main_progress_path)
    if main_progress.get("status") != "completed":
        raise ValueError(f"Main nested run is not completed: {main_progress}")
    main_config = read_json(main_run / "experiment_config.json")
    if main_config.get("scoring_mode") != "independent_per_dimension" or main_config.get("mapping_mode") != "with_mapping":
        raise ValueError("E6 expects the frozen main Qwen run (independent scoring with mapping)")
    if main_config.get("dataset_sha256") != dataset_hash or main_config.get("split_sha256") != split_hash:
        raise ValueError("Main run was produced from different data or splits")
    config_hash = main_progress.get("config_hash")
    if not config_hash:
        raise ValueError("Main progress lacks its configuration hash")

    cache_dir = (args.embedding_cache or (output_dir / "embedding_cache")).resolve()
    encoder_path = args.encoder_path.resolve() if args.encoder_path else None
    if encoder_path is not None and not encoder_path.is_dir():
        raise FileNotFoundError(f"Local BGE snapshot is missing: {encoder_path}")
    vectors, embedding_meta = get_embeddings(rows, dataset_hash, cache_dir, args.batch_size,
                                              args.device, encoder_path)
    embedding_by_id = {row["id"]: vectors[i] for i, row in enumerate(rows)}
    row_by_id = {row["id"]: row for row in rows}

    scorer = scoring_module or nested.load_scorer(HERE / "scorer.py")

    method_names = (["qwen", "llm_calibrated", "bge_knn", "bge_ridge", "learned_qwen_bge"] if args.primary_only else
                    ["qwen", "char_tfidf_knn", "char_tfidf_ridge", "bge_knn", "bge_ridge",
                     "learned_qwen_bge", "learned_char_bge"])

    output_config = {"protocol": "semantic_nested_5_outer_x_4_inner_oof_v1",
                     "dataset_sha256": dataset_hash, "split_sha256": split_hash,
                     "main_run_config_hash": config_hash, "encoder_model_id": MODEL_ID,
                     "encoder_revision": MODEL_REVISION, "pooling": "CLS", "normalize_l2": True,
                     "baseline_knn_k": KNN_K, "baseline_ridge_alpha": RIDGE_ALPHA,
                     "fusion_weight_grid": nested.WEIGHT_GRID,
                     "methods": method_names, "component_ablations": not args.primary_only,
                     "scorer_sha256": (sha256_file(args.scorer.resolve()) if args.scorer else None),
                     "dimensions": list(DIMENSIONS), "device": args.device}
    output_config_hash = hashlib.sha256(json.dumps(output_config, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    if config_path.exists() and read_json(config_path).get("config_hash") != output_config_hash:
        raise ValueError("Output directory already belongs to a different E6 config; choose a new directory")
    write_json_atomic(config_path, {**output_config, "config_hash": output_config_hash,
                                    "embedding_metadata": embedding_meta})

    all_predictions: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    write_json_atomic(output_dir / "progress.json",
                      {"status": "running", "completed_outer_folds": 0,
                       "total_outer_folds": len(outer_folds), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                       "config_hash": output_config_hash})
    inner_counts: dict[str, int] = {}
    for outer in outer_folds:
        outer_train = [row for row in rows if assignment[row["id"]] != outer]
        outer_test = [row for row in rows if assignment[row["id"]] == outer]
        inner_features: dict[str, dict[str, Any]] = {}
        for inner in outer_folds:
            if inner == outer:
                continue
            inner_train = [row for row in outer_train if assignment[row["id"]] != inner]
            inner_test = [row for row in outer_train if assignment[row["id"]] == inner]
            sem = semantic_predictions(inner_train, inner_test, embedding_by_id)
            for row in inner_test:
                doc = load_role(main_run, outer, "inner_oof", row["id"], inner, config_hash)
                pred = doc["prediction"]
                base = doc["classical_predictions"]
                inner_features[row["id"]] = {
                    "gold": {dim: float(row[dim]) for dim in DIMENSIONS},
                    "qwen": {dim: float(pred["scores"][dim]["score"]) for dim in DIMENSIONS},
                    "char_tfidf_knn": base["char_tfidf_knn"],
                    "char_tfidf_ridge": base["char_tfidf_ridge"],
                    "bge_knn": sem[row["id"]]["bge_knn"],
                    "bge_ridge": sem[row["id"]]["bge_ridge"],
                }
        if set(inner_features) != {row["id"] for row in outer_train}:
            raise ValueError(f"Incomplete inner OOF features for outer fold {outer}")
        inner_counts[str(outer)] = len(inner_features)

        fitted: dict[str, dict[str, Any]] = {}
        ids_train = sorted(inner_features)
        for dim in DIMENSIONS:
            y = np.asarray([inner_features[sid]["gold"][dim] for sid in ids_train], dtype=np.float64)
            qwen_values = np.asarray([inner_features[sid]["qwen"][dim] for sid in ids_train], dtype=np.float64)
            llm_offset = max(-0.5, min(0.5, float(np.median(y - qwen_values))))
            qwen_calibrated = np.clip(qwen_values + llm_offset, 1.0, 5.0)
            qwen_bge = np.asarray([[inner_features[sid][key][dim]
                                    for key in ("qwen", "bge_knn", "bge_ridge")]
                                   for sid in ids_train], dtype=np.float64)
            w_q, off_q, diag_q = nested.fit_simplex(qwen_bge, y, with_offset=True)
            fitted[dim] = {"qwen_bge_components": ["qwen", "bge_knn", "bge_ridge"],
                           "qwen_bge_weights": w_q.tolist(), "qwen_bge_offset": off_q,
                           "llm_calibrated_offset": llm_offset,
                           "llm_calibrated_fit": {"train_mae": float(np.mean(np.abs(qwen_calibrated - y))),
                                                   "train_rmse": float(np.sqrt(np.mean((qwen_calibrated - y) ** 2)))},
                           "qwen_bge_fit": diag_q, "training_n": len(ids_train)}
            if not args.primary_only:
                char_bge = np.asarray([[inner_features[sid][key][dim]
                                        for key in ("char_tfidf_knn", "bge_knn", "bge_ridge")]
                                       for sid in ids_train], dtype=np.float64)
                w_c, off_c, diag_c = nested.fit_simplex(char_bge, y, with_offset=True)
                fitted[dim].update({"char_bge_components": ["char_tfidf_knn", "bge_knn", "bge_ridge"],
                                    "char_bge_weights": w_c.tolist(), "char_bge_offset": off_c,
                                    "char_bge_fit": diag_c})

        sem_test = semantic_predictions(outer_train, outer_test, embedding_by_id)
        fold_records: list[dict[str, Any]] = []
        for row in outer_test:
            doc = load_role(main_run, outer, "outer_test", row["id"], None, config_hash)
            pred, classical = doc["prediction"], doc["classical_predictions"]
            base = {"qwen": {dim: float(pred["scores"][dim]["score"]) for dim in DIMENSIONS},
                    "char_tfidf_knn": classical["char_tfidf_knn"],
                    "char_tfidf_ridge": classical["char_tfidf_ridge"],
                    "bge_knn": sem_test[row["id"]]["bge_knn"],
                    "bge_ridge": sem_test[row["id"]]["bge_ridge"]}
            scores = {key: {dim: float(base[key][dim]) for dim in DIMENSIONS}
                      for key in method_names
                      if key not in {"learned_qwen_bge", "learned_char_bge", "llm_calibrated"}}
            scores["learned_qwen_bge"] = {}
            if args.primary_only:
                scores["llm_calibrated"] = {}
            if not args.primary_only:
                scores["learned_char_bge"] = {}
            gold = {dim: float(row[dim]) for dim in DIMENSIONS}
            for dim in DIMENSIONS:
                spec = fitted[dim]
                xq = np.asarray([base[key][dim] for key in spec["qwen_bge_components"]])
                scores["learned_qwen_bge"][dim] = float(np.clip(xq @ np.asarray(spec["qwen_bge_weights"])
                                                                 + spec["qwen_bge_offset"], 1.0, 5.0))
                if args.primary_only:
                    scores["llm_calibrated"][dim] = float(np.clip(
                        base["qwen"][dim] + spec["llm_calibrated_offset"], 1.0, 5.0))
                if not args.primary_only:
                    xc = np.asarray([base[key][dim] for key in spec["char_bge_components"]])
                    scores["learned_char_bge"][dim] = float(np.clip(xc @ np.asarray(spec["char_bge_weights"])
                                                                     + spec["char_bge_offset"], 1.0, 5.0))
            fold_records.append({"sample_id": row["id"], "outer_fold": outer,
                                 "group_id": row.get("group_id"), "gold": gold,
                                 "base_predictions": base, "scores": scores})
        all_predictions.extend(fold_records)
        all_weights.append({"outer_fold": outer, "dimensions": fitted})
        write_json_atomic(output_dir / f"outer_{outer}_predictions.json",
                          {"outer_fold": outer, "n": len(fold_records),
                           "predictions": fold_records, "weights": fitted})
        write_json_atomic(output_dir / "progress.json",
                          {"status": "running", "completed_outer_folds": outer + 1,
                           "total_outer_folds": len(outer_folds), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                           "config_hash": output_config_hash})

    method_names = output_config["methods"]
    metrics, fold_metrics = [], []
    for dim in DIMENSIONS:
        gold = [float(item["gold"][dim]) for item in all_predictions]
        for method in method_names:
            values = [float(item["scores"][method][dim]) for item in all_predictions]
            metrics.append({"dimension": dim, "method": method,
                            **nested.model_metrics(scorer, gold, values)})
            for outer in outer_folds:
                subset = [item for item in all_predictions if item["outer_fold"] == outer]
                fold_metrics.append({"dimension": dim, "method": method, "outer_fold": outer,
                                     **nested.model_metrics(scorer,
                                         [float(item["gold"][dim]) for item in subset],
                                         [float(item["scores"][method][dim]) for item in subset])})
    macro = []
    for method in method_names:
        entries = [item for item in metrics if item["method"] == method]
        macro.append({"method": method,
                      **{metric: statistics.mean(float(item[metric]) for item in entries if item[metric] is not None)
                         for metric in ("mae", "rmse", "pearson", "spearman", "rounded_qwk")}})
    result = {"status": "completed", "protocol": output_config["protocol"],
              "run": {"n": len(all_predictions), "outer_folds": outer_folds,
                      "embedding_dimension": int(vectors.shape[1]), "model_id": MODEL_ID,
                      "model_revision": MODEL_REVISION, "main_run_config_hash": config_hash,
                      "dataset_sha256": dataset_hash, "split_sha256": split_hash,
                      "completed_at_utc": datetime.now(timezone.utc).isoformat()},
              "metrics": metrics, "macro_metrics": macro, "fold_metrics": fold_metrics,
              "weights": all_weights, "predictions": all_predictions,
              "integrity_validation": {"dataset_split_match": True,
                                       "total_outer_predictions": len(all_predictions),
                                       "inner_oof_predictions_per_outer": inner_counts,
                                       "fusion_fit_uses_inner_oof_only": True,
                                       "embedding_uses_text_only": True}}
    write_json_atomic(output_dir / "nested_semantic.json", result)
    write_json_atomic(output_dir / "nested_semantic_metrics.json",
                      {"run": result["run"], "macro_metrics": macro, "metrics": metrics,
                       "fold_metrics": fold_metrics, "weights": all_weights,
                       "integrity_validation": result["integrity_validation"]})
    write_json_atomic(output_dir / "progress.json",
                      {"status": "completed", "completed_outer_folds": len(outer_folds),
                       "total_outer_folds": len(outer_folds), "total_outer_predictions": len(all_predictions),
                       "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "config_hash": output_config_hash})
    print(json.dumps({"status": "completed", "macro_metrics": macro,
                      "integrity_validation": result["integrity_validation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
