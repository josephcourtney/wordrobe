import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wordrobe.cli import app

runner = CliRunner()


def test_cli_runs_successfully() -> None:
    result = runner.invoke(app)

    assert result.exit_code == 0


@pytest.mark.parametrize("entrypoint", ["console-script", "module"])
def test_installed_entrypoint_runs_successfully(entrypoint: str) -> None:
    if entrypoint == "console-script":
        executable = shutil.which(
            "wordrobe",
            path=str(Path(sys.executable).parent),
        )
        assert executable is not None
        command = [executable]
    else:
        command = [sys.executable, "-m", "wordrobe"]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
