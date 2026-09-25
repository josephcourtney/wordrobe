from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

import wordrobe.segment as segment_module
from wordrobe.cli import app

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_segment_command_uses_dksplit_backend() -> None:
    result = runner.invoke(app, ["segment", "isthisasnorql", "--boundary-model", "dksplit"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is this a snorql"


def test_decode_command_uses_dksplit_backend() -> None:
    result = runner.invoke(app, ["decode", "isthisasnorql", "--boundary-model", "dksplit"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "is this a snorql"


def test_convert_command_uses_dksplit_backend() -> None:
    result = runner.invoke(
        app,
        ["convert", "isthisasnorql", "--to", "snake_case", "--boundary-model", "dksplit"],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "is_this_a_snorql"


def test_neural_weight_zero_matches_heuristic_cli() -> None:
    heuristic = runner.invoke(app, ["decode", "nowhere"])
    neural_zero = runner.invoke(
        app,
        ["decode", "nowhere", "--boundary-model", "dksplit", "--neural-weight", "0"],
    )

    assert heuristic.exit_code == 0
    assert neural_zero.exit_code == 0
    assert neural_zero.stdout == heuristic.stdout


def test_negative_neural_weight_is_rejected_by_cli() -> None:
    result = runner.invoke(
        app,
        ["decode", "test", "--boundary-model", "dksplit", "--neural-weight", "-1"],
    )

    assert result.exit_code != 0


def test_cli_reports_missing_neural_extra_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = segment_module.import_module

    def import_without_neural(name: str):
        if name == "wordrobe._neural_scoring":
            msg = "simulated missing NumPy"
            raise ImportError(msg)
        return real_import_module(name)

    monkeypatch.setattr(segment_module, "import_module", import_without_neural)
    result = runner.invoke(app, ["decode", "test", "--boundary-model", "dksplit"])

    assert result.exit_code == 2
    assert "requires NumPy" in result.stderr
    assert "wordrobe[neural]" in result.stderr
