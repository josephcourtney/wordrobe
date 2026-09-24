import pytest

from wordrobe import (
    AmbiguousCaseError,
    Case,
    CaseError,
    InvalidCaseError,
    decode,
    encode,
    guess_case,
    is_reversible,
    possible_cases,
    translate,
)
from wordrobe.case import REVERSIBLE_CASES


WORDS = ["hello", "world", "example"]


@pytest.mark.parametrize("case", sorted(REVERSIBLE_CASES, key=lambda item: item.value))
def test_reversible_cases_round_trip(case: Case) -> None:
    encoded = encode(WORDS, case)

    assert is_reversible(case)
    assert decode(encoded, case) == WORDS
    assert case in possible_cases(encoded)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (Case.SNAKE, "hello_world_example"),
        (Case.SCREAMING_SNAKE, "HELLO_WORLD_EXAMPLE"),
        (Case.KEBAB, "hello-world-example"),
        (Case.SCREAMING_KEBAB, "HELLO-WORLD-EXAMPLE"),
        (Case.TRAIN, "Hello-World-Example"),
        (Case.DOT, "hello.world.example"),
        (Case.PATH, "hello/world/example"),
        (Case.LOWER, "hello world example"),
        (Case.UPPER, "HELLO WORLD EXAMPLE"),
        (Case.TITLE, "Hello World Example"),
        (Case.SENTENCE, "Hello world example"),
        (Case.CAMEL, "helloWorldExample"),
        (Case.PASCAL, "HelloWorldExample"),
        (Case.FLAT, "helloworldexample"),
        (Case.UPPER_FLAT, "HELLOWORLDEXAMPLE"),
    ],
)
def test_encode(case: Case, expected: str) -> None:
    assert encode(WORDS, case) == expected


def test_translate_between_reversible_and_display_cases() -> None:
    assert translate("hello_world_example", Case.SNAKE, Case.CAMEL) == "helloWorldExample"


def test_decode_rejects_nonreversible_cases() -> None:
    with pytest.raises(CaseError):
        decode("helloWorld", Case.CAMEL)


def test_decode_rejects_noncanonical_input() -> None:
    with pytest.raises(InvalidCaseError):
        decode("Hello_World", Case.SNAKE)


def test_possible_cases_exposes_single_word_ambiguity() -> None:
    candidates = possible_cases("hello")

    assert Case.SNAKE in candidates
    assert Case.KEBAB in candidates
    assert Case.CAMEL in candidates
    assert Case.FLAT in candidates


def test_guess_case_returns_unique_match() -> None:
    assert guess_case("hello_world") is Case.SNAKE


def test_guess_case_returns_none_for_ambiguity() -> None:
    assert guess_case("hello") is None


def test_guess_case_can_raise_for_ambiguity() -> None:
    with pytest.raises(AmbiguousCaseError) as exc_info:
        guess_case("hello", should_raise=True)

    assert Case.CAMEL in exc_info.value.candidates


def test_guess_case_can_raise_for_invalid_text() -> None:
    with pytest.raises(InvalidCaseError):
        guess_case("hello_world-example", should_raise=True)


def test_encode_rejects_separator_inside_component_word() -> None:
    with pytest.raises(ValueError, match="not ASCII alphanumeric"):
        encode(["new_york", "city"], Case.SNAKE)
