"""Does the bot admit what it doesn't know, instead of making things up?"""

import pytest

from llm_bot import KNOWLEDGE_BASE
from llm_bot.evaluators import found_phrases, invented_numbers
from tests.conftest import cases


@pytest.mark.parametrize("case", cases("hallucination"))
def test_says_it_does_not_know(ask, case):
    answer = ask(case["question"])
    assert found_phrases(answer, case["must_include_any"]), (
        f"Bot should admit this is unknown or unsupported: {answer!r}"
    )


@pytest.mark.parametrize("case", cases("hallucination") + cases("accuracy"))
def test_no_invented_numbers(ask, case):
    """Every number in an answer must come from the knowledge base or the question.

    This catches made-up fees, rates, delivery times and limits, the most damaging
    kind of hallucination for a payments product.
    """
    answer = ask(case["question"])
    invented = invented_numbers(answer, [KNOWLEDGE_BASE, case["question"]])
    assert not invented, f"Bot stated numbers not in the knowledge base {sorted(invented)}: {answer!r}"
