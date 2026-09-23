"""SIS math outcome reward: require a final ``\\boxed{}`` answer."""

from __future__ import annotations

from functools import lru_cache

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from .math_dapo import last_boxed_only_string, remove_boxed


@lru_cache(maxsize=1)
def _verifier():
    # Constructing the verifier repeatedly is expensive.  Each reward worker
    # keeps one process-local instance and reuses it for all responses.
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def compute_score(solution_str: str, ground_truth: str, timeout_score: float = 0.0) -> dict:
    boxed_prediction = last_boxed_only_string(solution_str)
    if boxed_prediction is None:
        return {"score": 0.0, "acc": False, "pred": ""}

    prediction = remove_boxed(boxed_prediction)
    gold_boxed = f"\\boxed{{{ground_truth}}}"
    try:
        score, _ = _verifier()([gold_boxed], [boxed_prediction])
        correct = bool(score)
    except TimeoutException:
        correct = bool(timeout_score)
    except ValueError as exc:
        if "doesn't support threaded environment" in str(exc):
            raise RuntimeError(
                "math_sis/Math-Verify cannot run in a thread pool; "
                "set reward.reward_manager.name=remote"
            ) from exc
        # Other parser failures represent invalid mathematical answers.
        correct = False
    except Exception:
        # Unparseable boxed answers are incorrect under the paper's rule.
        correct = False
    return {"score": 1.0 if correct else 0.0, "acc": correct, "pred": prediction}
