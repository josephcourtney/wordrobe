from typer.testing import CliRunner

from wordrobe.cli import app

runner = CliRunner()


def test_segment_command_prints_segmented_text() -> None:
    result = runner.invoke(app, ["segment", "word2number"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "word 2 number"


def test_encode_command_prints_encoded_text() -> None:
    result = runner.invoke(app, ["encode", "--case", "camelCase", "hello", "world"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "helloWorld"


def test_decode_command_prints_component_words() -> None:
    result = runner.invoke(app, ["decode", "--case", "snake_case", "hello_world"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello world"


def test_convert_command_translates_case() -> None:
    result = runner.invoke(
        app,
        ["convert", "--from", "snake_case", "--to", "PascalCase", "hello_world"],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "HelloWorld"


def test_guess_command_prints_unique_case() -> None:
    result = runner.invoke(app, ["guess", "hello_world"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "snake_case"


def test_guess_all_prints_compatible_cases() -> None:
    result = runner.invoke(app, ["guess", "--all", "hello"])

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


def test_guess_command_rejects_ambiguous_text() -> None:
    result = runner.invoke(app, ["guess", "hello"])

    assert result.exit_code == 2
    assert "ambiguous" in result.stderr


def test_decode_command_rejects_noncanonical_text() -> None:
    result = runner.invoke(app, ["decode", "--case", "snake_case", "Hello_World"])

    assert result.exit_code == 2
    assert "does not match any supported case" in result.stderr
