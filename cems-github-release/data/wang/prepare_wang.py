"""Build a three-trait benchmark from exported public individual ratings."""
import collections
import csv
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent
DIMS = ("familiarity", "meaningfulness", "figurativeness")


def normalize(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = ROOT / "raw"
    derived = ROOT / "derived"
    derived.mkdir(exist_ok=True)
    wb = openpyxl.load_workbook(raw / "Table_1.xlsx", read_only=True, data_only=True)
    materials = []
    for row in wb["Sheet1"].iter_rows(values_only=True):
        if isinstance(row[0], (int, float)) and 1 <= row[0] <= 120 and all(isinstance(t, str) and t.strip() for t in row[1:5]):
            for condition, text in zip(("novel_metaphor", "novel_simile", "literal", "anomalous"), row[1:5]):
                materials.append({"family": int(row[0]), "condition": condition, "text": normalize(text)})
    wb.close()
    ratings = {dim: {} for dim in DIMS}
    invalid = collections.Counter()
    participant_counts = collections.Counter()
    provenance = []
    for dim, files in {"familiarity": [6, 7, 8], "meaningfulness": [9, 10, 11, 12],
                       "figurativeness": [13, 14, 15]}.items():
        for number in files:
            path = raw / f"Data_Sheet_{number}.xlsx"
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheet = wb.worksheets[0]
            rows = list(sheet.iter_rows(max_col=180, values_only=True))
            columns = {j: normalize(re.sub(r"^\s*\d+\s*[、.．]\s*", "", str(value)))
                       for j, value in enumerate(rows[0]) if j > 2 and value is not None
                       and re.match(r"^\s*\d+\s*[、.．]", str(value))}
            subjects = [row for row in rows[2:] if row[0] is not None and re.match(r"^\s*\d+", str(row[0]))]
            participant_counts[dim] += len(subjects)
            for col, text in columns.items():
                if text in ratings[dim]:
                    raise ValueError(f"Duplicate text within {dim}: {text}")
                values = []
                for row in subjects:
                    value = row[col]
                    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and 1 <= value <= 7:
                        values.append(float(value))
                    else:
                        invalid[dim] += 1
                ratings[dim][text] = values
            provenance.append({"file": path.name, "sha256": sha(path), "sheet": sheet.title,
                               "participant_rows": len(subjects), "item_columns": len(columns)})
            wb.close()
    output, excluded = [], []
    for item in materials:
        if item["condition"] == "anomalous":
            continue
        missing = [dim for dim in DIMS if not ratings[dim].get(item["text"])]
        if missing:
            excluded.append({**item, "missing_dimensions": missing})
            continue
        row = {"id": f"wang_{item['family']:03d}_{item['condition']}", "text": item["text"],
               "group_id": f"wang_family_{item['family']:03d}", "sentence_type": item["condition"],
               "source_sheet": "Wang2022/Table_1.xlsx", "sensory_domain": "general_sentence",
               "part_of_speech": "sentence"}
        for dim in DIMS:
            values = ratings[dim][item["text"]]
            native = statistics.mean(values)
            row[dim] = 1 + (native - 1) * (4 / 6)
            row[dim + "_native7"] = native
            row[dim + "_n"] = len(values)
            row[dim + "_sd7"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(row)
    groups = sorted({row["group_id"] for row in output})
    random.Random(20261022).shuffle(groups)
    group_folds = {group: index % 5 for index, group in enumerate(groups)}
    for row in output:
        row["fold"] = group_folds[row["group_id"]]
    assert len(output) == 356 and len(groups) == 120
    dataset = derived / "wang_items.csv"
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    splits = {"seed": 20261022, "grouping": "original critical-word family", "n_groups": len(groups),
              "assignment": {r["id"]: r["fold"] for r in output}, "group_assignment": group_folds}
    (derived / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"dataset": "Wang2022 public-export three-trait reconstruction", "n": len(output),
                "n_groups": len(groups), "dimensions": list(DIMS), "native_range": [1, 7], "model_range": [1, 5],
                "scale_mapping": "1+(native7-1)*4/6, fixed endpoints, no test-label fit",
                "aggregation": "arithmetic mean of exported participant ratings in 1..7",
                "exclusion": "drop out-of-range/nonfinite/nonnumeric cells; strict text join; do not impute missing traits",
                "qualification": "No additional inferred 3-SD rater rule. Derived means are not asserted identical to the paper's final norm means.",
                "exported_participant_rows": dict(participant_counts), "invalid_rating_cells": dict(invalid),
                "excluded_items": excluded, "source_files": provenance,
                "dataset_sha256": sha(dataset), "split_sha256": sha(derived / "splits.json"),
                "plan": {"outer_folds": 5, "inner_folds": 4, "anchor_count": 5,
                         "llm": "Qwen/Qwen3-14B-AWQ", "llm_revision": "55182c306332ddb9f47ade5de8d8fdb53b8bcb03",
                         "encoder": "BAAI/bge-small-zh-v1.5", "encoder_revision": "7999e1d3359715c523056ef9478215996d62a620",
                         "temperature": 0, "thinking": False, "mapping": True, "independent_traits": True,
                         "knn_k": 5, "ridge_alpha": 10, "simplex_step": 0.05, "bias_range": [-0.5, 0.5],
                         "variants": ["fixed_llm", "llm_calibrated", "bge_knn", "bge_ridge", "without_llm", "full_bge",
                                      "without_bge_knn", "without_bge_ridge", "without_bias"],
                         "primary_metric": "macro MAE", "bootstrap": "2000 paired family resamples, percentile interval",
                         "selection": "no model selection using outer-test outcomes", "gpu_count": 1,
                         "gpu_physical_index": 1, "role_predictions": len(output) * 5,
                         "score_calls_without_repairs": len(output) * 5 * 3}}
    (derived / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(output), "groups": len(groups), "conditions": dict(collections.Counter(r["sentence_type"] for r in output)),
                      "participants": dict(participant_counts), "invalid_cells": dict(invalid),
                      "roles": len(output) * 5, "score_calls": len(output) * 15}, ensure_ascii=False))


if __name__ == "__main__":
    main()
