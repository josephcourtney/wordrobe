from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from wordrobe._meta import metadata
from wordrobe.case import (
    CASE_PREFERENCE,
    Case,
    CaseError,
    guess_case,
    is_reversible,
    possible_cases,
)
from wordrobe.case import (
    decode as decode_case,
)
from wordrobe.case import (
    encode as encode_case,
)
from wordrobe.segment import DEFAULT_NEURAL_WEIGHT, BoundaryModel, WordSegmenter

CLI_DESCRIPTION = "Split, identify, and convert word casing."

app = typer.Typer(
    name=metadata.executable or metadata.name,
    help=CLI_DESCRIPTION,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "auto_envvar_prefix": encode_case([metadata.name], Case.SCREAMING_SNAKE),
    },
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=True,
    pretty_exceptions_short=True,
)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(metadata.version)
        raise typer.Exit()


def _recover_words(
    text: str,
    from_case: Case | None = None,
    boundary_model: BoundaryModel = BoundaryModel.HEURISTIC,
    neural_weight: float = DEFAULT_NEURAL_WEIGHT,
) -> list[str]:
    """Recover semantic words, using exact source boundaries when available."""
    if from_case is not None and is_reversible(from_case):
        return decode_case(text, from_case)

    if from_case is not None and from_case not in possible_cases(text):
        msg = f"{text!r} is not canonical {from_case.value}."
        raise CaseError(msg)

    segmenter = WordSegmenter(boundary_model=boundary_model, neural_weight=neural_weight)
    return [word.lower() for word in segmenter.segment(text)]


def _echo_words(
    text: str,
    from_case: Case | None = None,
    boundary_model: BoundaryModel = BoundaryModel.HEURISTIC,
    neural_weight: float = DEFAULT_NEURAL_WEIGHT,
) -> None:
    try:
        words = _recover_words(text, from_case, boundary_model, neural_weight)
    except (CaseError, ImportError, ValueError) as exc:
        _fail(str(exc))

    typer.echo(" ".join(words))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the Wordrobe version and exit.",
        ),
    ] = False,
) -> None:
    """Split, identify, and convert word casing."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("split")
def split_command(
    text: Annotated[str, typer.Argument(help="Text whose component words should be recovered.")],
    from_case: Annotated[
        Case | None,
        typer.Option(
            "--from",
            "-f",
            help="Require this source case; reversible cases split exactly.",
        ),
    ] = None,
    boundary_model: Annotated[
        BoundaryModel,
        typer.Option(
            "--boundary-model",
            "-b",
            help="Boundary evidence model for inferred word boundaries.",
            rich_help_panel="Boundary recovery",
        ),
    ] = BoundaryModel.HEURISTIC,
    neural_weight: Annotated[
        float,
        typer.Option(
            "--neural-weight",
            min=0.0,
            help="Weight of neural CRF evidence when the DKSplit model is selected.",
            rich_help_panel="Boundary recovery",
        ),
    ] = DEFAULT_NEURAL_WEIGHT,
) -> None:
    """Recover component words from text."""
    _echo_words(text, from_case, boundary_model, neural_weight)


@app.command("join")
def join_command(
    words: Annotated[list[str], typer.Argument(help="Component words to format.")],
    to_case: Annotated[
        Case,
        typer.Option("--to", "-t", help="Target case convention."),
    ],
) -> None:
    """Format component words in a case convention."""
    try:
        value = encode_case(words, to_case)
    except (TypeError, ValueError) as exc:
        _fail(str(exc))

    typer.echo(value)


@app.command("convert")
def convert_command(
    text: Annotated[str, typer.Argument(help="Text to convert.")],
    to_case: Annotated[
        Case,
        typer.Option("--to", "-t", help="Target case convention."),
    ],
    from_case: Annotated[
        Case | None,
        typer.Option(
            "--from",
            "-f",
            help="Require this source case; omit to recover boundaries automatically.",
        ),
    ] = None,
    boundary_model: Annotated[
        BoundaryModel,
        typer.Option(
            "--boundary-model",
            "-b",
            help="Boundary evidence model for inferred word boundaries.",
            rich_help_panel="Boundary recovery",
        ),
    ] = BoundaryModel.HEURISTIC,
    neural_weight: Annotated[
        float,
        typer.Option(
            "--neural-weight",
            min=0.0,
            help="Weight of neural CRF evidence when the DKSplit model is selected.",
            rich_help_panel="Boundary recovery",
        ),
    ] = DEFAULT_NEURAL_WEIGHT,
) -> None:
    """Recover component words and format them in another case convention."""
    try:
        words = _recover_words(text, from_case, boundary_model, neural_weight)
        value = encode_case(words, to_case)
    except (CaseError, ImportError, TypeError, ValueError) as exc:
        _fail(str(exc))

    typer.echo(value)


@app.command("case")
def case_command(
    text: Annotated[str, typer.Argument(help="Text whose case convention should be identified.")],
    *,
    all_matches: Annotated[
        bool,
        typer.Option("--all", "-a", help="Print every compatible case convention."),
    ] = False,
) -> None:
    """Identify the case convention used by text."""
    if all_matches:
        candidates = possible_cases(text)
        if not candidates:
            _fail(f"{text!r} does not match any supported case.")
        for candidate in candidates:
            typer.echo(candidate.value)
        return

    try:
        candidate = guess_case(text, should_raise=True)
    except CaseError as exc:
        _fail(str(exc))

    if candidate is None:
        msg = "strict case identification returned no result"
        raise AssertionError(msg)
    typer.echo(candidate.value)


@app.command("cases")
def cases_command() -> None:
    """List supported case conventions and their boundary semantics."""
    width = max(len(case.value) for case in CASE_PREFERENCE)
    for candidate in CASE_PREFERENCE:
        boundary_kind = "explicit boundaries" if is_reversible(candidate) else "inferred boundaries"
        typer.echo(f"{candidate.value:<{width}}  {boundary_kind}")


if __name__ == "__main__":
    app()
