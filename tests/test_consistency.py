"""Does the bot give the same facts when asked the same question again?

LLMs are not deterministic. A support bot that quotes 3.99 one time and 2.99 the next
is a real production bug, so the same question is asked several times (--repeat).
"""

import pytest

from llm_bot.evaluators import numbers_in
from tests.conftest import cases


@pytest.mark.parametrize("case", cases("consistency"))
def test_same_numbers_every_time(ask, repeat, case):
    answers = [ask(case["question"]) for _ in range(repeat)]
    number_sets = [numbers_in(a) for a in answers]

    assert all(n == number_sets[0] for n in number_sets), (
        "Answers gave different numbers across runs:\n" + "\n".join(f"- {a}" for a in answers)
    )
