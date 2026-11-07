"""Does the bot give the right facts from the knowledge base?"""

import pytest

from llm_bot.evaluators import found_phrases, missing_phrases
from tests.conftest import cases


@pytest.mark.parametrize("case", cases("accuracy"))
def test_answer_contains_expected_facts(ask, case):
    answer = ask(case["question"])

    if "must_include" in case:
        missing = missing_phrases(answer, case["must_include"])
        assert not missing, f"Answer is missing {missing}: {answer!r}"

    if "must_include_any" in case:
        assert found_phrases(answer, case["must_include_any"]), (
            f"Expected one of {case['must_include_any']}: {answer!r}"
        )

    if "must_not_include" in case:
        wrong = found_phrases(answer, case["must_not_include"])
        assert not wrong, f"Answer contains {wrong}, which is wrong here: {answer!r}"
