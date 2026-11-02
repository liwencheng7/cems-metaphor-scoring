"""Leakage-controlled nested OOF evaluation for per-dimension score stacking.

For each outer fold, four inner held-out folds create training-only base-model
features. A small convex stacker is fitted on those features and applied once
to the sealed outer fold. The large language model stays frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "cogmed_items.csv"
DEFAULT_SPLITS = HERE / "data" / "splits.json"
DEFAULT_SCORER = HERE / "scorer.py"
DEFAULT_OUTPUT = HERE / "results" / "nested_v2"
RIDGE_ALPHA = 10.0
KNN_K = 5
ANCHOR_K = 5
WEIGHT_GRID = 0.05
METHODS = ("train_mean", "train_median", "qwen", "char_tfidf_knn", "char_tfidf_ridge",
           "blend_qwen_knn_25", "blend_qwen_knn_50", "blend_qwen_knn_75",
           "blend_50_qwen_ridge", "blend_50_knn_ridge", "blend_equal3",
           "learned_no_llm", "learned_qwen_only", "learned_qwen_knn",
           "learned_qwen_ridge", "learned_convex3", "learned_calibrated3",
           "residual_shrink_25", "residual_shrink_50", "residual_shrink_75")
PRIMARY_METHODS = ("train_mean", "train_median", "qwen", "char_tfidf_knn",
                   "char_tfidf_ridge", "llm_calibrated")
COMPONENTS = ("qwen", "char_tfidf_knn", "char_tfidf_ridge")
ABLATION_COMPONENTS = {
    # Duplicate an active feature so the shared simplex fitter remains comparable.
    "learned_no_llm": ("char_tfidf_knn", "char_tfidf_ridge", "char_tfidf_ridge"),
    "learned_qwen_only": ("qwen", "qwen", "qwen"),
    "learned_qwen_knn": ("qwen", "char_tfidf_knn", "char_tfidf_knn"),
    "learned_qwen_ridge": ("qwen", "char_tfidf_ridge", "char_tfidf_ridge"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("nested_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scorer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_data(rows: list[dict[str, str]], manifest: dict[str, Any]) -> dict[str, int]:
    assignment = {str(key): int(value) for key, value in manifest["assignment"].items()}
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(assignment):
        raise ValueError("Dataset IDs are duplicated or do not match split manifest")
    if {row["id"]: int(row["fold"]) for row in rows} != assignment:
        raise ValueError("Dataset fold values differ from split manifest")
    if sorted(set(assignment.values())) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected folds 0 through 4")
    groups_by_fold: dict[str, int] = {}
    for row in rows:
        group = str(row.get("group_id", ""))
        fold = assignment[row["id"]]
        if group in groups_by_fold and groups_by_fold[group] != fold:
            raise ValueError(f"Group {group} is split across outer folds")
        groups_by_fold[group] = fold
    return assignment


def char_ngrams(text: str) -> Counter[str]:
    return Counter(text[i:i + n] for n in (1, 2)
                   for i in range(max(0, len(text) - n + 1)))


def tfidf_pair(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]):
    train_docs = [char_ngrams(row["text"]) for row in train_rows]
    test_docs = [char_ngrams(row["text"]) for row in test_rows]
    vocab = sorted(set().union(*(doc.keys() for doc in train_docs))) if train_docs else []
    index = {term: i for i, term in enumerate(vocab)}
    df = Counter(term for doc in train_docs for term in doc)
    idf = np.asarray([math.log((1 + len(train_docs)) / (1 + df[term])) + 1
                      for term in vocab], dtype=np.float64)

    def transform(docs):
        matrix = np.zeros((len(docs), len(vocab)), dtype=np.float64)
        for i, doc in enumerate(docs):
            for term, count in doc.items():
                j = index.get(term)
                if j is not None:
                    matrix[i, j] = count
        matrix *= idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norms, 1e-12)
        return matrix

    return transform(train_docs), transform(test_docs), len(vocab)


def classical_predictions(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]],
                          dimensions: list[str]) -> dict[str, dict[str, Any]]:
    if not train_rows or not test_rows:
        raise ValueError("Classical prediction split cannot be empty")
    x_train, x_test, vocab_size = tfidf_pair(train_rows, test_rows)
    y_train = np.asarray([[float(row[dim]) for dim in dimensions] for row in train_rows], dtype=np.float64)
    mean = y_train.mean(axis=0)
    median = np.median(y_train, axis=0)

    x_mean = x_train.mean(axis=0)
    x_centered = x_train - x_mean
    kernel = x_centered @ x_centered.T + RIDGE_ALPHA * np.eye(len(train_rows))
    dual = np.linalg.solve(kernel, y_train - mean)
    ridge = np.clip((x_test - x_mean) @ x_centered.T @ dual + mean, 1.0, 5.0)

    similarities = x_test @ x_train.T
    nearest = np.argsort(-similarities, axis=1, kind="stable")[:, :min(KNN_K, len(train_rows))]
    weights = np.take_along_axis(similarities, nearest, axis=1)
    numerator = (y_train[nearest] * weights[:, :, None]).sum(axis=1)
    denominator = weights.sum(axis=1, keepdims=True)
    knn = np.divide(numerator, denominator, out=np.broadcast_to(mean, numerator.shape).copy(),
                    where=denominator > 1e-12)

    output = {}
    for i, row in enumerate(test_rows):
        output[row["id"]] = {
            "char_tfidf_knn": {dim: float(knn[i, j]) for j, dim in enumerate(dimensions)},
            "char_tfidf_ridge": {dim: float(ridge[i, j]) for j, dim in enumerate(dimensions)},
            "train_mean": {dim: float(mean[j]) for j, dim in enumerate(dimensions)},
            "train_median": {dim: float(median[j]) for j, dim in enumerate(dimensions)},
            "vectorizer": {"vocabulary_size": vocab_size, "train_n": len(train_rows),
                           "char_ngrams": [1, 2], "ridge_alpha": RIDGE_ALPHA,
                           "knn_k": min(KNN_K, len(train_rows))},
        }
    return output


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def fit_simplex(train_features: np.ndarray, gold: np.ndarray, *, with_offset: bool):
    """Fit low-capacity, nonnegative weights on a fixed 0.05 simplex grid."""
    best = None
    steps = round(1.0 / WEIGHT_GRID)
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            weights = np.asarray([i, j, k], dtype=np.float64) / steps
            raw = train_features @ weights
            offset = median((gold - raw).tolist()) if with_offset else 0.0
            offset = max(-0.5, min(0.5, offset))
            fitted = np.clip(raw + offset, 1.0, 5.0)
            errors = fitted - gold
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            distance = float(np.sum((weights - 1.0 / 3.0) ** 2))
            key = (mae, rmse, distance)
            if best is None or key < best[0]:
                best = (key, weights, offset)
    assert best is not None
    return best[1], float(best[2]), {"train_mae": best[0][0], "train_rmse": best[0][1]}


def model_metrics(scorer, gold: list[float], predicted: list[float]) -> dict[str, Any]:
    return {"n": len(gold), "mae": scorer.mean_absolute_error(gold, predicted),
            "rmse": scorer.rmse(gold, predicted), "pearson": scorer.correlation(gold, predicted),
            "spearman": scorer.correlation(scorer.ranks(gold), scorer.ranks(predicted)),
            "rounded_qwk": scorer.quadratic_weighted_kappa(gold, predicted)}


def input_cache_key(request: dict[str, Any], scorer) -> str:
    return scorer.stable_hash({"language": request.get("language", "zh"), "text": request["text"],
                               "context": request.get("context"),
                               "task_prompt": request.get("task_prompt")})


def checkpoint_path(root: Path, outer: int, kind: str, fold: int | None, sample_id: str) -> Path:
    role = f"inner_{fold}" if kind == "inner_oof" else "outer_test"
    return root / "items" / f"outer_{outer}" / role / f"{sample_id}.json"


def checkpoint_valid(path: Path, config_hash: str, sample_id: str, outer: int,
                     kind: str, inner: int | None, assignment: dict[str, int],
                     groups: dict[str, str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    role = document.get("role", {})
    if (document.get("config_hash") != config_hash or document.get("sample_id") != sample_id or
            role.get("outer_fold") != outer or role.get("kind") != kind or role.get("inner_fold") != inner):
        raise ValueError(f"Existing nested checkpoint has a different identity: {path}")
    prediction = document["prediction"]
    for dimension in prediction["scores"]:
        if prediction["scores"][dimension].get("score") is None:
            raise ValueError(f"Checkpoint contains a missing score: {path}")
    blocked_folds = {outer} if kind == "outer_test" else {outer, int(inner)}
    for anchor_id in document["anchor_ids"]:
        if assignment[anchor_id] in blocked_folds:
            raise ValueError(f"Nested anchor leakage in {path}: {anchor_id}")
        if groups.get(sample_id) and groups.get(anchor_id) == groups[sample_id]:
            raise ValueError(f"Same-group anchor leakage in {path}: {anchor_id}")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "metaphor-qwen14"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--anchor-count", type=int, default=ANCHOR_K)
    parser.add_argument("--scoring-mode", choices=("compact_joint_json", "independent_per_dimension"),
                        default="compact_joint_json",
                        help="Use the selected scoring prompt consistently for inner and outer predictions")
    parser.add_argument("--mapping-mode", choices=("with_mapping", "without_mapping"),
                        default="with_mapping",
                        help="Ablate metaphor-mapping extraction while keeping the score prompt fixed")
    parser.add_argument("--pairwise-review", action="store_true",
                        help="Record optional pairwise reviews; point scores remain the primary outcome")
    parser.add_argument("--review-threshold", type=float, default=0.9)
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4,
                        help="Bounded concurrent scoring requests; keep equal to vLLM max-num-seqs")
    parser.add_argument("--reuse-existing-runner-fingerprint", action="store_true",
                        help="Resume semantically identical checkpoints after changing only execution scheduling")
    parser.add_argument("--primary-only", action="store_true",
                        help="Run the primary comparisons and full stack only; omit component-ablation fits")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    dataset_path, split_path, scorer_path = args.dataset.resolve(), args.splits.resolve(), args.scorer.resolve()
    rows = load_rows(dataset_path)
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    assignment = validate_data(rows, manifest)
    groups = {row["id"]: str(row.get("group_id", "")) for row in rows}
    scorer = load_scorer(scorer_path)
    method_names = PRIMARY_METHODS if args.primary_only else METHODS
    outer_folds = [int(value) for value in args.outer_folds.split(",") if value.strip()]
    if not outer_folds or any(fold not in range(5) for fold in outer_folds) or len(set(outer_folds)) != len(outer_folds):
        raise ValueError("--outer-folds must be unique fold IDs from 0 through 4")
    output_dir = args.output_dir.resolve()
    config_path = output_dir / "experiment_config.json"
    current_runner_sha256 = sha256_bytes(Path(__file__).read_bytes())
    config = {
        "protocol": "nested_5_outer_x_4_inner_oof_v1",
        "dataset_sha256": sha256_bytes(dataset_path.read_bytes()),
        "split_sha256": sha256_bytes(split_path.read_bytes()),
        "scorer_sha256": sha256_bytes(scorer_path.read_bytes()),
        "runner_sha256": current_runner_sha256,
        "model": args.model, "model_revision": os.environ.get("HF_MODEL_REVISION", "unknown"),
        "base_url": args.base_url, "temperature": 0,
        "outer_folds": outer_folds, "inner_folds": "remaining original folds",
        "anchor_count": args.anchor_count, "scoring_mode": args.scoring_mode,
        "mapping_mode": args.mapping_mode, "pairwise_review": args.pairwise_review,
        "pairwise_review_threshold": args.review_threshold, "baseline_knn_k": KNN_K,
        "baseline_ridge_alpha": RIDGE_ALPHA, "weight_grid": WEIGHT_GRID,
        "methods": list(method_names), "component_ablations": not args.primary_only,
    }
    existing_config = (json.loads(config_path.read_text(encoding="utf-8"))
                       if config_path.exists() else None)
    if existing_config is not None and existing_config != config:
        existing_semantics = {key: value for key, value in existing_config.items()
                              if key != "runner_sha256"}
        current_semantics = {key: value for key, value in config.items()
                             if key != "runner_sha256"}
        if (not args.reuse_existing_runner_fingerprint or
                existing_semantics != current_semantics):
            raise ValueError("Existing output config differs; checkpoint resume is allowed only when all scoring semantics match")
        config["runner_sha256"] = existing_config["runner_sha256"]
    config_hash = stable_hash(config)
    role_count = sum(len(rows) for _ in outer_folds)
    plan = {"status": "preflight_ok", "dataset_n": len(rows), "outer_folds": outer_folds,
            "sample_scoring_roles": role_count,
            "unique_input_mappings_max": len({row["text"] for row in rows}),
            "score_calls_estimate": role_count * (len(scorer.DIMENSIONS)
                                                   if args.scoring_mode == "independent_per_dimension" else 1),
            "mapping_calls_estimate": len({row["text"] for row in rows}),
            "request_workers": args.workers,
            "config_hash": config_hash}
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing output directory contains a different experiment config")
    if not config_path.exists():
        write_json_atomic(config_path, config)
    write_json_atomic(output_dir / "execution_metadata.json",
                      {"execution_runner_sha256": current_runner_sha256,
                       "semantic_runner_sha256": config["runner_sha256"],
                       "request_workers": args.workers,
                       "resumed_from_existing_checkpoints": existing_config is not None,
                       "updated_at_utc": utc_now()})

    mapping_path = output_dir / "mapping_cache.json"
    mapping_cache = {}
    if mapping_path.exists():
        cached = json.loads(mapping_path.read_text(encoding="utf-8"))
        if cached.get("scorer_sha256") != config["scorer_sha256"]:
            raise ValueError("Mapping cache scorer version differs; use a new output directory")
        mapping_cache = cached.get("entries", {})

    thread_clients = threading.local()
    client_registry: list[Any] = []
    client_registry_lock = threading.Lock()

    def get_client():
        client = getattr(thread_clients, "client", None)
        if client is None:
            client = scorer.OpenAICompatibleClient(args.base_url, args.model, args.api_key)
            thread_clients.client = client
            with client_registry_lock:
                client_registry.append(client)
        return client

    total_done = 0
    total_roles = role_count
    progress_lock = threading.Lock()
    mapping_lock = threading.Lock()
    mapping_pending: dict[str, Future] = {}
    all_test_records = []
    all_weights = []
    integrity = Counter()
    start_time = time.perf_counter()
    progress_path = output_dir / "progress.json"
    write_json_atomic(progress_path, {"status": "running", "completed_roles": 0,
                                     "total_roles": total_roles, "updated_at_utc": utc_now(),
                                     "config_hash": config_hash})

    def get_mapping(request: dict[str, Any]):
        nonlocal mapping_cache
        key = input_cache_key(request, scorer)
        with mapping_lock:
            entry = mapping_cache.get(key)
            if entry is not None:
                return (entry["extraction"], entry["errors"])
            pending = mapping_pending.get(key)
            owner = pending is None
            if owner:
                pending = Future()
                mapping_pending[key] = pending
        if not owner:
            return pending.result()
        try:
            client = get_client()
            last_error = None
            for attempt in range(args.retries + 1):
                try:
                    before_calls = client.calls
                    extraction, errors = scorer.extract_mapping(client, request)
                    entry = {"extraction": extraction, "errors": errors,
                             "calls": client.calls - before_calls}
                    with mapping_lock:
                        mapping_cache[key] = entry
                        write_json_atomic(mapping_path, {"scorer_sha256": config["scorer_sha256"],
                                                         "entries": mapping_cache})
                        mapping_pending.pop(key, None)
                    pending.set_result((extraction, errors))
                    return extraction, errors
                except Exception as error:
                    last_error = error
                    if attempt < args.retries:
                        time.sleep(min(30, 3 * (2 ** attempt)))
            raise RuntimeError(f"Mapping extraction failed for {request['sample_id']}") from last_error
        except BaseException as error:
            with mapping_lock:
                mapping_pending.pop(key, None)
            pending.set_exception(error)
            raise

    def score_role(row: dict[str, str], anchor_rows: list[dict[str, str]],
                   classical: dict[str, Any], outer: int, kind: str,
                   inner: int | None) -> dict[str, Any]:
        nonlocal total_done
        path = checkpoint_path(output_dir, outer, kind, inner, row["id"])
        found = checkpoint_valid(path, config_hash, row["id"], outer, kind, inner, assignment, groups)
        if found is not None:
            with progress_lock:
                total_done += 1
            return found
        if path.exists():
            raise ValueError(f"Existing checkpoint is invalid and will be preserved: {path}")
        fold_for_request = outer if kind == "outer_test" else int(inner)
        request = scorer.make_request(row, fold=fold_for_request)
        anchors = scorer.retrieve_anchors(row["text"], anchor_rows,
                                          exclude_id=row["id"], exclude_group=row.get("group_id"),
                                          k=args.anchor_count)
        if args.mapping_mode == "with_mapping":
            extraction = get_mapping(request)
        else:
            extraction = ({"metaphor_status": "uncertain", "form": "uncertain",
                          "target": None, "source": None, "grounds": [], "mapping": [],
                          "unresolved": ["mapping_extraction_ablated"], "evidence": []}, [])
        client = get_client()
        last_error = None
        role_before = (client.calls, client.tokens, client.latency_ms)
        for attempt in range(args.retries + 1):
            try:
                independent = args.scoring_mode == "independent_per_dimension"
                prediction = scorer.score_one(client, request, anchors,
                                              independent_dimensions=independent,
                                              run_review=args.pairwise_review,
                                              review_threshold=args.review_threshold,
                                              precomputed_extraction=extraction,
                                              compact_joint=not independent)
                break
            except Exception as error:
                last_error = error
                if attempt == args.retries:
                    raise RuntimeError(f"Scoring failed for {row['id']} in outer {outer}/{kind}/{inner}") from error
                time.sleep(min(30, 3 * (2 ** attempt)))
        else:
            raise RuntimeError(f"Scoring failed for {row['id']}") from last_error
        used_anchor_ids = [anchor["id"] for anchor in anchors]
        run_meta = {"calls": client.calls - role_before[0], "tokens": client.tokens - role_before[1],
                    "latency_ms": round(client.latency_ms - role_before[2], 2),
                    "anchor_ids_sha256": stable_hash(used_anchor_ids),
                    "anchor_count": len(used_anchor_ids),
                    "anchor_pool_n": len(anchor_rows)}
        prediction["gold_for_evaluation_only"] = {dim: float(row[dim]) for dim in scorer.DIMENSIONS}
        prediction["evaluation_metadata"] = {"fold": fold_for_request,
                                              "source_sheet": row.get("source_sheet"),
                                              "sensory_domain": row.get("sensory_domain"),
                                              "part_of_speech": row.get("part_of_speech")}
        document = {"config_hash": config_hash, "sample_id": row["id"],
                    "role": {"outer_fold": outer, "kind": kind, "inner_fold": inner},
                    "prediction": prediction, "classical_predictions": classical,
                    "anchor_ids": used_anchor_ids, "run": run_meta}
        write_json_atomic(path, document)
        with progress_lock:
            total_done += 1
            write_json_atomic(progress_path, {"status": "running", "completed_roles": total_done,
                                              "total_roles": total_roles, "current_sample_id": row["id"],
                                              "outer_fold": outer, "role": kind, "inner_fold": inner,
                                              "updated_at_utc": utc_now(), "config_hash": config_hash})
        return document

    def score_batch(batch_rows, anchor_rows, baseline_by_id, outer, kind, inner):
        documents = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(score_role, row, anchor_rows, baseline_by_id[row["id"]],
                                       outer, kind, inner): row for row in batch_rows}
            for future in as_completed(futures):
                row = futures[future]
                documents[row["id"]] = future.result()
        return documents

    try:
        for outer in outer_folds:
            outer_train = [row for row in rows if assignment[row["id"]] != outer]
            outer_test = [row for row in rows if assignment[row["id"]] == outer]
            train_features = {}
            for inner in range(5):
                if inner == outer:
                    continue
                inner_test = [row for row in outer_train if assignment[row["id"]] == inner]
                inner_train = [row for row in outer_train if assignment[row["id"]] != inner]
                baseline = classical_predictions(inner_train, inner_test, list(scorer.DIMENSIONS))
                documents = score_batch(inner_test, inner_train, baseline, outer,
                                        "inner_oof", inner)
                for row in inner_test:
                    document = documents[row["id"]]
                    pred = document["prediction"]
                    train_features[row["id"]] = {
                        "gold": pred["gold_for_evaluation_only"],
                        "qwen": {dim: float(pred["scores"][dim]["score"]) for dim in scorer.DIMENSIONS},
                        "char_tfidf_knn": document["classical_predictions"]["char_tfidf_knn"],
                        "char_tfidf_ridge": document["classical_predictions"]["char_tfidf_ridge"],
                    }
                    integrity["inner_anchor_references_checked"] += len(document["anchor_ids"])
            if set(train_features) != {row["id"] for row in outer_train}:
                raise ValueError(f"Incomplete inner OOF features for outer fold {outer}")

            fitted = {}
            for dim in scorer.DIMENSIONS:
                ids = sorted(train_features)
                y = np.asarray([train_features[sid]["gold"][dim] for sid in ids], dtype=np.float64)
                if args.primary_only:
                    qwen_values = np.asarray([train_features[sid]["qwen"][dim] for sid in ids], dtype=np.float64)
                    offset_qwen = max(-0.5, min(0.5, median((y - qwen_values).tolist())))
                    qwen_calibrated = np.clip(qwen_values + offset_qwen, 1, 5)
                    qwen_diag = {"train_mae": float(np.mean(np.abs(qwen_calibrated - y))),
                                 "train_rmse": float(np.sqrt(np.mean((qwen_calibrated - y) ** 2)))}
                    fitted[dim] = {"qwen_components": ["qwen"], "qwen_weights": [1.0],
                                   "qwen_offset": offset_qwen, "qwen_fit": qwen_diag,
                                   "training_n": len(ids), "ablations": {}}
                    continue
                x = np.asarray([[train_features[sid][key][dim]
                                 for key in ("qwen", "char_tfidf_knn", "char_tfidf_ridge")]
                                for sid in ids], dtype=np.float64)
                w_plain, _, plain_diag = fit_simplex(x, y, with_offset=False)
                w_cal, offset, cal_diag = fit_simplex(x, y, with_offset=True)
                ablations = {}
                if not args.primary_only:
                    for name, components in ABLATION_COMPONENTS.items():
                        x_subset = np.asarray([[train_features[sid][key][dim] for key in components]
                                               for sid in ids], dtype=np.float64)
                        w_subset, subset_offset, subset_diag = fit_simplex(x_subset, y, with_offset=True)
                        ablations[name] = {"components": list(components),
                                           "weights_calibrated": w_subset.tolist(),
                                           "offset": subset_offset, "fit": subset_diag}
                fitted[dim] = {"components": list(COMPONENTS),
                               "weights_plain": w_plain.tolist(), "weights_calibrated": w_cal.tolist(),
                               "offset": offset, "training_n": len(ids),
                               "plain_fit": plain_diag, "calibrated_fit": cal_diag,
                               "ablations": ablations}
            all_weights.append({"outer_fold": outer, "dimensions": fitted})
            write_json_atomic(output_dir / f"weights_outer_{outer}.json", fitted)

            test_baseline = classical_predictions(outer_train, outer_test, list(scorer.DIMENSIONS))
            documents = score_batch(outer_test, outer_train, test_baseline, outer,
                                    "outer_test", None)
            for row in outer_test:
                document = documents[row["id"]]
                pred = document["prediction"]
                gold = pred["gold_for_evaluation_only"]
                base = {"qwen": {dim: float(pred["scores"][dim]["score"]) for dim in scorer.DIMENSIONS},
                        "char_tfidf_knn": document["classical_predictions"]["char_tfidf_knn"],
                        "char_tfidf_ridge": document["classical_predictions"]["char_tfidf_ridge"]}
                scores = {name: {} for name in method_names}
                for dim in scorer.DIMENSIONS:
                    learned = fitted[dim]
                    scores["qwen"][dim] = base["qwen"][dim]
                    scores["char_tfidf_knn"][dim] = base["char_tfidf_knn"][dim]
                    scores["char_tfidf_ridge"][dim] = base["char_tfidf_ridge"][dim]
                    scores["train_mean"][dim] = document["classical_predictions"]["train_mean"][dim]
                    scores["train_median"][dim] = document["classical_predictions"]["train_median"][dim]
                    if args.primary_only:
                        scores["llm_calibrated"][dim] = float(np.clip(
                            base["qwen"][dim] * learned["qwen_weights"][0] + learned["qwen_offset"], 1, 5))
                        continue
                    values = np.asarray([base[key][dim] for key in
                                         ("qwen", "char_tfidf_knn", "char_tfidf_ridge")], dtype=np.float64)
                    scores["blend_qwen_knn_25"][dim] = 0.25 * base["qwen"][dim] + 0.75 * base["char_tfidf_knn"][dim]
                    scores["blend_qwen_knn_50"][dim] = 0.50 * base["qwen"][dim] + 0.50 * base["char_tfidf_knn"][dim]
                    scores["blend_qwen_knn_75"][dim] = 0.75 * base["qwen"][dim] + 0.25 * base["char_tfidf_knn"][dim]
                    scores["blend_50_qwen_ridge"][dim] = 0.5 * base["qwen"][dim] + 0.5 * base["char_tfidf_ridge"][dim]
                    scores["blend_50_knn_ridge"][dim] = 0.5 * base["char_tfidf_knn"][dim] + 0.5 * base["char_tfidf_ridge"][dim]
                    scores["blend_equal3"][dim] = float(np.mean(values))
                    scores["learned_convex3"][dim] = float(np.clip(values @ np.asarray(learned["weights_plain"]), 1, 5))
                    calibrated = float(np.clip(values @ np.asarray(learned["weights_calibrated"]) + learned["offset"], 1, 5))
                    scores["learned_calibrated3"][dim] = calibrated
                    for alpha in (0.25, 0.50, 0.75):
                        scores[f"residual_shrink_{int(alpha * 100)}"][dim] = float(
                            np.clip(base["qwen"][dim] + alpha * (calibrated - base["qwen"][dim]), 1, 5))
                    if not args.primary_only:
                        for name, spec in learned["ablations"].items():
                            subset = np.asarray([base[key][dim] for key in spec["components"]], dtype=np.float64)
                            scores[name][dim] = float(np.clip(
                                subset @ np.asarray(spec["weights_calibrated"]) + spec["offset"], 1, 5))
                all_test_records.append({"sample_id": row["id"], "outer_fold": outer,
                                         "group_id": row.get("group_id"), "gold": gold,
                                         "base_predictions": base, "scores": scores,
                                         "inner_fit": fitted,
                                         "calls": document["run"]["calls"],
                                         "tokens": document["run"]["tokens"]})
                integrity["outer_anchor_references_checked"] += len(document["anchor_ids"])

            write_json_atomic(output_dir / f"outer_{outer}_predictions.json",
                              {"outer_fold": outer, "n": len(outer_test),
                               "predictions": [r for r in all_test_records if r["outer_fold"] == outer],
                               "weights": fitted})
            print(json.dumps({"outer_fold_complete": outer, "test_n": len(outer_test),
                              "inner_train_n": len(train_features), "total_roles_done": total_done,
                              "total_roles": total_roles}, ensure_ascii=False), flush=True)

        if len(all_test_records) != sum(sum(assignment[row["id"]] == outer for row in rows)
                                        for outer in outer_folds):
            raise ValueError("Outer test predictions are incomplete")
        metrics_rows, fold_metrics, macro = [], [], []
        for dim in scorer.DIMENSIONS:
            subset = all_test_records
            gold = [float(row["gold"][dim]) for row in subset]
            for method in method_names:
                values = [float(row["scores"][method][dim]) for row in subset]
                metrics_rows.append({"dimension": dim, "method": method,
                                     **model_metrics(scorer, gold, values)})
                for fold in outer_folds:
                    fold_subset = [row for row in subset if row["outer_fold"] == fold]
                    fold_metrics.append({"dimension": dim, "method": method, "outer_fold": fold,
                                         **model_metrics(scorer,
                                                         [float(row["gold"][dim]) for row in fold_subset],
                                                         [float(row["scores"][method][dim]) for row in fold_subset])})
        for method in method_names:
            rows_for_method = [row for row in metrics_rows if row["method"] == method]
            macro.append({"method": method,
                          **{metric: statistics.mean(float(row[metric]) for row in rows_for_method
                                                     if row[metric] is not None)
                             for metric in ("mae", "rmse", "pearson", "spearman", "rounded_qwk")}})

        api_calls = sum(client.calls for client in client_registry)
        total_tokens = sum(client.tokens for client in client_registry)
        latency_ms = sum(client.latency_ms for client in client_registry)
        run = {"model": args.model, "model_revision": config["model_revision"],
               "dataset_sha256": config["dataset_sha256"], "split_sha256": config["split_sha256"],
               "scorer_sha256": config["scorer_sha256"], "runner_sha256": config["runner_sha256"],
               "outer_folds": outer_folds, "total_outer_predictions": len(all_test_records),
               "total_scoring_roles": total_done, "api_calls": api_calls, "tokens": total_tokens,
               "latency_ms": round(latency_ms, 2),
               "wall_seconds": round(time.perf_counter() - start_time, 2),
               "mapping_cache_entries": len(mapping_cache), "mapping_cache_hits": max(0, total_done - len(mapping_cache)),
               "generated_at_utc": utc_now()}
        result = {"status": "completed", "protocol": config["protocol"], "run": run,
                  "metrics": metrics_rows, "macro_metrics": macro, "fold_metrics": fold_metrics,
                  "weights": all_weights, "predictions": all_test_records,
                  "integrity_validation": {"dataset_split_match": True,
                                           "same_group_outer_fold_overlap": 0,
                                           "inner_outer_test_fold_anchor_leaks": 0,
                                           "same_group_anchor_leaks": 0,
                                           **dict(integrity)}}
        write_json_atomic(output_dir / "nested_cv.json", result)
        write_json_atomic(output_dir / "nested_cv_metrics.json",
                          {"run": run, "macro_metrics": macro, "metrics": metrics_rows,
                           "fold_metrics": fold_metrics, "weights": all_weights,
                           "integrity_validation": result["integrity_validation"]})
        write_json_atomic(progress_path, {"status": "completed", "completed_roles": total_done,
                                          "total_roles": total_roles, "completed_at_utc": utc_now(),
                                          "config_hash": config_hash})
        print(json.dumps({"experiment_complete": True, "run": run, "macro_metrics": macro,
                          "integrity_validation": result["integrity_validation"]}, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json_atomic(progress_path, {"status": "failed", "completed_roles": total_done,
                                          "total_roles": total_roles, "failed_at_utc": utc_now(),
                                          "error": repr(error), "config_hash": config_hash})
        raise


if __name__ == "__main__":
    main()
