import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from chat import ask


def test_ask_returns_string():
    """The ask() function should always return a non-empty string."""
    result = ask("What is 2 + 2? Answer in one word.")
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"Response: {result}")


def test_ask_is_relevant():
    """The answer to a capital city question should mention the city."""
    result = ask("What is the capital of France? One word only.")
    assert "Paris" in result
    print(f"Response: {result}")


def test_additional_cases():
    result = ask("Which country is Bali from?")
    assert "Indonesia" in result
    print(f"Response: {result}")
