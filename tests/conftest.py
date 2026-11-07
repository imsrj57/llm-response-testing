from pathlib import Path

import pytest
import yaml

from llm_bot import get_provider

CASES = yaml.safe_load(
    (Path(__file__).resolve().parent.parent / "data" / "test_cases.yaml").read_text(encoding="utf-8")
)


def cases(group: str):
    """Turn a group from test_cases.yaml into pytest parameters, using each case id as the test name."""
    return [pytest.param(case, id=case["id"]) for case in CASES[group]]


def pytest_addoption(parser):
    parser.addoption(
        "--repeat", type=int, default=3,
        help="How many times to ask each question in the consistency tests (default 3).",
    )


@pytest.fixture(scope="session")
def bot():
    return get_provider()


@pytest.fixture(scope="session")
def repeat(request):
    return request.config.getoption("--repeat")


@pytest.fixture
def ask(bot, request):
    """Ask the bot a question and keep the Q&A so it shows up in the HTML report."""

    def _ask(question: str) -> str:
        answer = bot.ask(question)
        request.node.user_properties.append(("question", question))
        request.node.user_properties.append(("answer", answer))
        print(f"\nQ: {question}\nA: {answer}")
        return answer

    return _ask


def pytest_report_header(config):
    import os

    return f"LLM provider: {os.getenv('LLM_PROVIDER', 'mock')}"
