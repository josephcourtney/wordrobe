# Changelog

All notable user-visible changes to `wordrobe` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- CLI commands for case encoding, strict decoding, case conversion, and case recognition.
- `wordrobe guess --all` for deterministic enumeration of compatible case conventions.
- Documentation for custom segmentation vocabulary, blocked words, unknown-word scoring, frequency files, and lossless spans.

### Changed

- CLI subcommand option parsing now delegates options to the selected subcommand instead of treating them as root options.

## [0.1.0] - 2026-09-24

### Added

- Strict case encoding, reversible decoding, conversion, compatibility detection, and ambiguity-aware case guessing.
- Low-resource word segmentation using common vocabulary, system dictionaries, optional frequency data, custom and blocked words, unknown-word scoring, punctuation boundaries, capitalization hints, numeric boundaries, and lossless source spans.
- `wordrobe segment` command-line interface.
