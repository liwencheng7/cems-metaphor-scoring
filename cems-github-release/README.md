# CEMS: Continuous Scoring of Chinese Metaphor-Related Texts

CEMS predicts human mean ratings for Chinese metaphor-related expressions. It combines a rubric-guided language-model scorer with a nearest-neighbor predictor and ridge regression over frozen BGE embeddings. Fusion weights are fitted per dimension using predictions from inner folds of the outer training set.

The paper evaluates five-fold grouped cross-validation on COGMED (671 expressions, seven dimensions) and Wang (356 sentences, three dimensions). On the original scales, macro MAE is 0.265 for CEMS and 0.292 for the LLM alone on COGMED, and 0.549 and 0.602 respectively on Wang. COGMED rounded QWK is 0.537 for CEMS and 0.544 for the LLM alone. See the paper and `results/` for the full comparisons.

## Contents

- `paper/`: the paper, LaTeX sources, bibliography and editable SVG figures. Open `paper/paper.pdf` for the compiled paper.
- `runtime/`: scoring, grouped nested evaluation, BGE predictors and adapted AutoSCORE/RULERS experiment code.
- `data/wang/`: Wang preprocessing script and group assignment. `runtime/data/splits.json` records the COGMED split summary.
- `results/`: aggregate per-dimension and macro metrics, plus backbone and adapted-baseline comparisons.

## Data and reproduction

Obtain COGMED from [Huang et al. (2024)](https://www.nature.com/articles/s41599-024-03593-2/) and the Wang materials and rating exports from [Wang (2022)](https://www.frontiersin.org/articles/10.3389/fpsyg.2022.922722/full). The expected paths and preprocessing are described in [data/README.md](data/README.md). Dataset redistribution terms should be checked with the original sources.

The reported results use frozen Qwen3-14B-AWQ and BAAI/bge-small-zh-v1.5, with Qwen3-8B and Llama-3.1-8B as additional scorers. Model weights, raw datasets, individual responses, and item-level model outputs are not in this package. An OpenAI-compatible model endpoint and the source datasets are required to run scoring. Aggregate metrics alone do not support recalculating paired confidence intervals. The published scripts have not been validated end-to-end in a clean environment.

Install the dependencies listed in `requirements.txt`, supply the data, and inspect command options with:

```bash
python runtime/run_nested_cv.py --help
python runtime/evaluate_semantic_nested.py --help
```

For the paper's frozen COGMED scorer, pass `--scorer runtime/scorer_remote_paper_20261018.py` to `run_nested_cv.py`. The BGE evaluation consumes the per-role checkpoints produced by the nested run; its `--main-run` must point to that run. The adapted-baseline runner uses `CEMS_RUN_DIR` for outputs and `CEMS_API_URL` and `CEMS_MODEL` for the model endpoint; defaults are a local endpoint on port 8029 and model name `cems-supp-qwen14b`. The runner saves request/response audits containing input texts, so keep those outputs private until their redistribution rights are established.

AutoSCORE was proposed for educational response scoring; the repository adapts its extraction–scoring design to metaphor ratings. The RULERS adaptation evaluates the direct continuous score and omits the original calibration stage. See the paper for the different output formats and evaluation conditions.

## License

A license for project-authored code and manuscript has not yet been selected. The source datasets, model weights and conference style files have their own terms.
