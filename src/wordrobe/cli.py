from __future__ import annotations

import typer

from wordrobe._meta import metadata
from wordrobe.case import Case, encode
from wordrobe.segment import WordSegmenter

ENTRYPOINT_NAME = ""
HELP_HEADER = f"{metadata.executable}: {metadata.description}"

app = typer.Typer(
    name=metadata.executable or metadata.name,
    help=metadata.description,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "auto_envvar_prefix": encode([metadata.name], Case.SCREAMING_SNAKE),
        "allow_interspersed_args": True,
    },
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=True,
    pretty_exceptions_short=True,
)


@app.callback(invoke_without_command=True)
def main() -> None:
    """Wordrobe."""


@app.command()
def segment(text: str) -> None:
    wordlist = None

    segmenter = WordSegmenter(wordlist=wordlist)

    typer.echo(segmenter.segment_string(text))


if __name__ == "__main__":
    raise SystemExit(main())
