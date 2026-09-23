from concurrent.futures import ThreadPoolExecutor

from verl.utils.reward_score.math_sis import compute_score


def test_requires_boxed_answer():
    assert compute_score("Answer: 34", "34")["score"] == 0.0
    assert compute_score(r"Therefore, \boxed{34}.", "34")["score"] == 1.0


def test_accepts_equivalent_boxed_math():
    result = compute_score(r"Therefore, \boxed{0.5}.", r"\frac{1}{2}")
    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["pred"] == "0.5"


def test_uses_last_boxed_answer():
    assert compute_score(r"First \boxed{1}, finally \boxed{2}.", "2")["score"] == 1.0
    assert compute_score(r"First \boxed{2}, finally \boxed{1}.", "2")["score"] == 0.0


def test_threaded_math_verify_fails_loudly():
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(compute_score, r"Therefore, \boxed{34}.", "34")
        try:
            future.result()
        except RuntimeError as exc:
            assert "reward.reward_manager.name=remote" in str(exc)
        else:
            raise AssertionError("threaded Math-Verify must not silently turn a correct answer into score 0")
