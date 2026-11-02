# Data preparation

The repository contains group assignments and preprocessing code, but does not redistribute source materials, participant rating sheets, item-level predictions or model responses.

## COGMED

Download the materials from the COGMED paper and prepare `runtime/data/cogmed_items.csv` with the columns used by `runtime/scorer.py`, including `id`, `text`, `group_id`, `fold` and the seven rating dimensions. The original experiment used 671 expressions and preserved the released direction of the `difficulty_raw` field. `runtime/data/splits.json` records the split summary; the fold assignments must agree with the prepared item table.

## Wang

Download the supplementary files for Wang (2022) into `data/wang/raw/`, retaining names such as `Table_1.xlsx` and `Data_Sheet_6.xlsx` through `Data_Sheet_15.xlsx`. Run:

```bash
python data/wang/prepare_wang.py
```

The script writes `data/wang/derived/wang_items.csv` and its metadata. Text matching and score aggregation retain 356 matched items from 120 families. It reconstructs means from exported ratings; equivalence to the original study's final norms has not been established. See the paper appendix for exclusions and invalid cells.

The raw data and generated item table are excluded from Git by `.gitignore`. Verify the original sources' licenses before redistributing them.
