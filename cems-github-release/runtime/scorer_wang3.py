"""Use the preserved COGMED scorer with three overlapping Wang constructs."""
import hashlib
from pathlib import Path

_source = Path(__file__).with_name("scorer_remote_paper_20261018.py")
_source_bytes = _source.read_bytes()
SCORER_ORIGIN_SHA256 = hashlib.sha256(_source_bytes).hexdigest()
EXPECTED_SCORER_ORIGIN_SHA256 = "5f8fc73975fe5e865779aee53f7befb6860d4dc27c634039e28bf708ae2e21a7"
if SCORER_ORIGIN_SHA256 != EXPECTED_SCORER_ORIGIN_SHA256:
    raise RuntimeError("The frozen COGMED scorer source has changed")
exec(compile(_source_bytes, str(_source), "exec"), globals())
DIMENSIONS = {key: value for key, value in DIMENSIONS.items()
              if key in ("familiarity", "meaningfulness", "figurativeness")}
_original_client = OpenAICompatibleClient
_original_score_one = score_one


class OpenAICompatibleClient(_original_client):
    def __init__(self, base_url, model, api_key="EMPTY", timeout=240, chat_template_kwargs=None):
        super().__init__(base_url, model, api_key, timeout,
                         {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs)


def make_request(row, *, fold):
    return {"sample_id": row["id"], "language": "zh", "evaluation_unit": "sentence", "text": row["text"],
            "task_prompt": "预测该中文句子的熟悉度、有意义度和比喻性人评均分。分数统一为1–5；原始1–7人评按固定端点线性换算。评分时不得看到待评句子的人评标签。",
            "context": None, "audience": "普通话母语者", "rubric_id": "wang2022_3traits_affine_1_5_v1",
            "evaluation_fold": fold}


def score_one(client, request, anchors, **kwargs):
    result = _original_score_one(client, request, anchors, **kwargs)
    result["task"] = "wang2022_three_trait_norm_prediction"
    for dimension, item in result["scores"].items():
        item["scale_id"] = f"WANG2022_{dimension}_1_5_affine_from_1_7"
        item["scale_transfer"] = "fixed_endpoint_affine_1_7_to_1_5"
    return result
