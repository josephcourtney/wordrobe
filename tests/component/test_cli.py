import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wordrobe._meta import metadata
from wordrobe.cli import app

runner = CliRunner()


def test_cli_without_command_shows_help() -> None:
    result = runner.invoke(app)

    assert result.exit_code == 0
    assert "Split, identify, and convert word casing." in result.stdout
    for command in ("split", "join", "convert", "case", "cases"):
        assert command in result.stdout


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == metadata.version


def test_split_recovers_words() -> None:
    result = runner.invoke(app, ["split", "isthisacamel"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is this a camel"


def test_split_from_reversible_case_is_exact() -> None:
    result = runner.invoke(app, ["split", "hello_world", "--from", "snake_case"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello world"


def test_join_formats_words() -> None:
    result = runner.invoke(app, ["join", "hello", "world", "--to", "camelCase"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "helloWorld"


def test_convert_recovers_and_formats_words() -> None:
    result = runner.invoke(app, ["convert", "isThisACamel", "--to", "snake_case"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is_this_a_camel"


def test_convert_from_reversible_case_is_exact() -> None:
    result = runner.invoke(
        app,
        ["convert", "hello_world", "--from", "snake_case", "--to", "PascalCase"],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "HelloWorld"


def test_case_identifies_unique_case() -> None:
    result = runner.invoke(app, ["case", "hello_world"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "snake_case"


def test_case_all_prints_compatible_cases() -> None:
    result = runner.invoke(app, ["case", "hello", "--all"])

    assert result.exit_code == 0
    assert "snake_case" in result.stdout
    assert "camelCase" in result.stdout
    assert "flatcase" in result.stdout


def test_cases_lists_boundary_semantics() -> None:
    result = runner.invoke(app, ["cases"])

    assert result.exit_code == 0
    assert "snake_case" in result.stdout
    assert "explicit boundaries" in result.stdout
    assert "camelCase" in result.stdout
    assert "inferred boundaries" in result.stdout


@pytest.mark.parametrize("removed_command", ["encode", "decode", "guess", "segment"])
def test_removed_command_names_are_rejected(removed_command: str) -> None:
    result = runner.invoke(app, [removed_command])

    assert result.exit_code != 0


@pytest.mark.parametrize("entrypoint", ["console-script", "module", "cli-module"])
def test_installed_entrypoint_runs_successfully(entrypoint: str) -> None:
    if entrypoint == "console-script":
        executable = shutil.which(
            "wordrobe",
            path=str(Path(sys.executable).parent),
        )
        assert executable is not None
        command = [executable]
    elif entrypoint == "module":
        command = [sys.executable, "-m", "wordrobe"]
    else:
        command = [sys.executable, "-m", "wordrobe.cli"]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "split" in result.stdout
    assert "join" in result.stdout
