from typer.testing import CliRunner

from wordrobe.cli import app

runner = CliRunner()


def test_segment_command_prints_segmented_text() -> None:
    result = runner.invoke(app, ["segment", "word2number"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "word 2 number"
