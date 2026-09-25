from __future__ import annotations

import pytest

from scripts.evaluate_hybrid_segmentation import CASES, Case
from wordrobe.segment import DEFAULT_NEURAL_WEIGHT


def _case_param(case: Case):
    marks = ()
    if case.xfail:
        marks = (pytest.mark.xfail(reason="known unresolved lexical ambiguity"),)
    return pytest.param(case, id=case.text, marks=marks)


@pytest.mark.parametrize("case", [_case_param(case) for case in CASES])
def test_default_dksplit_backend_adversarial_matrix(tmp_path, case: Case) -> None:
    wordlist = tmp_path / "missing-dictionary"
    segmenter = case.make_segmenter(wordlist, DEFAULT_NEURAL_WEIGHT)

    assert tuple(segmenter.segment(case.text)) in case.accepted
