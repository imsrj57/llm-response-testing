"""When asked for JSON, is the answer valid JSON with the right values?

Other systems often consume LLM output directly, so a broken or wrong JSON quote
is as bad as an API returning the wrong payload.
"""

import json

import pytest
from jsonschema import validate

from llm_bot.evaluators import parse_json
from tests.conftest import cases

FEE_QUOTE_SCHEMA = {
    "type": "object",
    "required": ["amount", "currency", "fee", "total"],
    "properties": {
        "amount": {"type": "number"},
        "currency": {"type": "string", "enum": ["USD"]},
        "fee": {"type": "number", "minimum": 0},
        "total": {"type": "number"},
    },
}


@pytest.mark.parametrize("case", cases("structured_output"))
def test_fee_quote_is_valid_json(ask, case):
    answer = ask(case["question"])
    try:
        quote = parse_json(answer)
    except json.JSONDecodeError as error:
        pytest.fail(f"Answer is not valid JSON ({error}): {answer!r}")

    validate(instance=quote, schema=FEE_QUOTE_SCHEMA)

    for key, expected in case["expected"].items():
        actual = quote[key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected), f"{key}: expected {expected}, got {actual}"
        else:
            assert actual == expected, f"{key}: expected {expected}, got {actual}"

    assert quote["total"] == pytest.approx(quote["amount"] + quote["fee"]), "total should equal amount + fee"
