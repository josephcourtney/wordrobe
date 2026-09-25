from typer.testing import CliRunner

from wordrobe.cli import app

runner = CliRunner()


def test_split_recovers_words_across_numeric_boundaries() -> None:
    result = runner.invoke(app, ["split", "word2number"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "word 2 number"


def test_split_recovers_reversible_case_without_case_classification() -> None:
    result = runner.invoke(app, ["split", "hello_world"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello world"


def test_split_recovers_camel_boundaries() -> None:
    result = runner.invoke(app, ["split", "isThisACamel"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is this a camel"


def test_split_viterbi_segments_flat_text() -> None:
    result = runner.invoke(app, ["split", "isthisacamel"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is this a camel"


def test_split_accepts_ambiguous_case_when_words_are_recoverable() -> None:
    result = runner.invoke(app, ["split", "hello"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"


def test_split_can_heuristically_recover_explicit_camel_case() -> None:
    result = runner.invoke(app, ["split", "isThisATest", "--from", "camelCase"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is this a test"


def test_join_prints_encoded_text() -> None:
    result = runner.invoke(app, ["join", "hello", "world", "--to", "camelCase"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "helloWorld"


def test_convert_translates_explicit_case() -> None:
    result = runner.invoke(
        app,
        ["convert", "hello_world", "--from", "snake_case", "--to", "PascalCase"],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "HelloWorld"


def test_convert_recovers_source_words_when_from_is_omitted() -> None:
    result = runner.invoke(app, ["convert", "isThisACamel", "--to", "snake_case"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is_this_a_camel"


def test_case_prints_unique_case() -> None:
    result = runner.invoke(app, ["case", "hello_world"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "snake_case"


def test_case_all_prints_compatible_cases_in_preference_order() -> None:
    result = runner.invoke(app, ["case", "hello", "--all"])

    assert result.exit_code == 0
    assert result.stdout.splitlines() == [
        "snake_case",
        "kebab-case",
        "dot.case",
        "path/case",
        "lower case",
        "camelCase",
        "flatcase",
    ]


def test_case_rejects_ambiguous_text_without_all() -> None:
    result = runner.invoke(app, ["case", "hello"])

    assert result.exit_code == 2
    assert "ambiguous" in result.stderr


def test_split_rejects_noncanonical_explicit_reversible_case() -> None:
    result = runner.invoke(app, ["split", "Hello_World", "--from", "snake_case"])

    assert result.exit_code == 2
    assert "does not match any supported case" in result.stderr


def test_split_rejects_noncanonical_explicit_inferred_case() -> None:
    result = runner.invoke(app, ["split", "HelloWorld", "--from", "camelCase"])

    assert result.exit_code == 2
    assert "not canonical camelCase" in result.stderr
