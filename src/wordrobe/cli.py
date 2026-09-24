from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from wordrobe._meta import metadata
from wordrobe.case import (
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


def _recover_words(text: str, case: Case | None = None) -> list[str]:
    """Recover semantic words using exact boundaries when requested, otherwise all available evidence."""
    if case is not None and is_reversible(case):
        return decode_case(text, case)

    if case is not None and case not in possible_cases(text):
        msg = f"{text!r} is not canonical {case.value}."
        raise CaseError(msg)

    return [word.lower() for word in WordSegmenter().segment(text)]


def _echo_words(text: str, case: Case | None = None) -> None:
    try:
        words = _recover_words(text, case)
    except CaseError as exc:
        _fail(str(exc))

    typer.echo(" ".join(words))


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
    text: Annotated[str, typer.Argument(help="Text to recover component words from.")],
    case: Annotated[
        Case | None,
        typer.Option("--case", "-c", help="Require this source case instead of automatic boundary recovery."),
    ] = None,
) -> None:
    """Recover semantic component words from text."""
    _echo_words(text, case)


@app.command("convert")
def convert_command(
    text: Annotated[str, typer.Argument(help="Text to convert.")],
    to_case: Annotated[Case, typer.Option("--to", help="Target case.")],
    from_case: Annotated[
        Case | None,
        typer.Option("--from", help="Require this source case instead of automatic boundary recovery."),
    ] = None,
) -> None:
    """Recover component words and encode them in another case convention."""
    try:
        words = _recover_words(text, from_case)
        value = encode_case(words, to_case)
    except (CaseError, TypeError, ValueError) as exc:
        _fail(str(exc))

    typer.echo(value)


@app.command("guess")
def guess_command(
    text: Annotated[str, typer.Argument(help="Text whose case should be identified.")],
    *,
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


@app.command("segment", hidden=True)
def segment_command(text: Annotated[str, typer.Argument(help="Text to recover component words from.")]) -> None:
    """Compatibility alias for `decode`."""
    _echo_words(text)


if __name__ == "__main__":
    app()
