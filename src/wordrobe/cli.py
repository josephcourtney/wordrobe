from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from wordrobe._meta import metadata
from wordrobe.case import (
    Case,
    CaseError,
    decode as decode_case,
    encode as encode_case,
    guess_case,
    possible_cases,
    translate,
)
from wordrobe.segment import WordSegmenter

ENTRYPOINT_NAME = ""
HELP_HEADER = f"{metadata.executable}: {metadata.description}"

app = typer.Typer(
    name=metadata.executable or metadata.name,
    help=metadata.description,
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


@app.callback(invoke_without_command=True)
def main() -> None:
    """Wordrobe."""


@app.command("encode")
def encode_command(
    words: Annotated[list[str], typer.Argument(help="Component words to encode.")],
    case: Annotated[Case, typer.Option("--case", "-c", help="Target case.")],
) -> None:
    """Encode component words in a case convention."""
    try:
        value = encode_case(words, case)
    except (TypeError, ValueError) as exc:
        _fail(str(exc))

    typer.echo(value)


@app.command("decode")
def decode_command(
    text: Annotated[str, typer.Argument(help="Canonical text to decode.")],
    case: Annotated[Case, typer.Option("--case", "-c", help="Source case.")],
) -> None:
    """Decode canonical text from a reversible case convention."""
    try:
        words = decode_case(text, case)
    except CaseError as exc:
        _fail(str(exc))

    typer.echo(" ".join(words))


@app.command("convert")
def convert_command(
    text: Annotated[str, typer.Argument(help="Canonical text to convert.")],
    from_case: Annotated[Case, typer.Option("--from", help="Source case.")],
    to_case: Annotated[Case, typer.Option("--to", help="Target case.")],
) -> None:
    """Convert canonical text between case conventions."""
    try:
        value = translate(text, from_case, to_case)
    except CaseError as exc:
        _fail(str(exc))

    typer.echo(value)


@app.command("guess")
def guess_command(
    text: Annotated[str, typer.Argument(help="Text whose case should be identified.")],
    all_matches: Annotated[
        bool,
        typer.Option("--all", help="Print every compatible case instead of requiring a unique match."),
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
        msg = "strict case guessing returned no result"
        raise AssertionError(msg)
    typer.echo(candidate.value)


@app.command()
def segment(text: str) -> None:
    """Heuristically recover word boundaries from text."""
    segmenter = WordSegmenter()
    typer.echo(segmenter.segment_string(text))


if __name__ == "__main__":
    app()
