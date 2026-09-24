"""Package metadata accessors."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution

_DEFAULT_DISTRIBUTION_NAME = __package__
_DEFAULT_EXECUTABLE_NAME = _DEFAULT_DISTRIBUTION_NAME

DISTRIBUTION_NAME = "wordrobe"


@dataclass(frozen=True, slots=True)
class Metadata:
    """Installed package metadata used by interfaces."""

    name: str
    version: str
    description: str
    executable: str | None


def _load_metadata() -> Metadata:
    """Load package version and console-script metadata.

    Returns
    -------
    Metadata
    """
    try:
        dist = distribution(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return Metadata(
            name=DISTRIBUTION_NAME,
            version="0.0.0",
            description="",
            executable=None,
        )

    executable = next(
        (entry_point.name for entry_point in dist.entry_points if entry_point.group == "console_scripts"),
        None,
    )

    return Metadata(
        name=dist.metadata["Name"],
        version=dist.version,
        description=dist.metadata.get("Summary", ""),
        executable=executable,
    )


metadata = _load_metadata()

__all__ = ["Metadata", "metadata"]
