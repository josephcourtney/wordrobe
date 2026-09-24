# Changelog

All notable user-visible changes to `wordrobe` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- CLI commands for case encoding, decoding, case conversion, and case recognition.
- `wordrobe guess --all` for deterministic enumeration of compatible case conventions.
- Documentation for custom segmentation vocabulary, blocked words, unknown-word scoring, frequency files, and lossless spans.

### Changed

- `wordrobe decode` now performs general word recovery when `--case` is omitted instead of requiring a unique case classification first.
- `wordrobe convert` now performs the same automatic source-word recovery when `--from` is omitted.
- `wordrobe segment` is retained as a hidden compatibility alias for `wordrobe decode` rather than exposing a separate recovery policy.
- Unknown-word Viterbi costs now grow linearly with token length and penalize very short unknown fragments, preventing long unseparated strings from winning merely because they were treated as one unknown token.
- Ranked fallback vocabulary duplicates now retain their earliest (best) rank.
- Unknown candidates may use the configured `max_word_length` even when the loaded dictionary contains only shorter words.
- CLI subcommand option parsing now delegates options to the selected subcommand instead of treating them as root options.

## [0.1.0] - 2026-09-24

### Added

- Strict case encoding, reversible decoding, conversion, compatibility detection, and ambiguity-aware case guessing.
- Low-resource word segmentation using common vocabulary, system dictionaries, optional frequency data, custom and blocked words, unknown-word scoring, punctuation boundaries, capitalization hints, numeric boundaries, and lossless source spans.
- `wordrobe segment` command-line interface.
