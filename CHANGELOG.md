# Changelog

All notable user-visible changes to `wordrobe` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A task-oriented CLI built around `wordrobe split`, `wordrobe join`, `wordrobe convert`, `wordrobe case`, and `wordrobe cases`.
- `wordrobe --version` and useful top-level help when `wordrobe` is invoked without a subcommand.
- `wordrobe case --all` for deterministic enumeration of compatible case conventions.
- `wordrobe cases` for listing supported conventions and whether their boundaries are explicit or inferred.
- Documentation for custom segmentation vocabulary, blocked words, unknown-word scoring, frequency files, and lossless spans.
- Optional `wordrobe[neural]` support with a DKSplit-derived NumPy boundary model that can contribute CRF boundary evidence to the existing lexical Viterbi search.
- `--boundary-model` and `--neural-weight` options for inferred CLI word boundaries.

### Changed

- The CLI now uses directional `--from` and `--to` options consistently: `split` accepts an optional source convention, `join` requires a target convention, and `convert` composes both operations.
- The former `encode`, `decode`, `guess`, and `segment` commands have been removed rather than retained as compatibility aliases.
- `wordrobe split` performs general word recovery when `--from` is omitted; reversible source cases split exactly when `--from` is supplied.
- `wordrobe convert` performs the same automatic source-word recovery when `--from` is omitted.
- Unknown-word Viterbi costs now use a calibrated linear character cost plus an explicit short-fragment penalty, allowing common short words to split from unknown neighbors without routinely fragmenting technical identifiers.
- Viterbi segmentation now includes an explicit penalty for unsupported inferred boundaries, minimal local sequence state for adjacent singleton words, and a contextual article-before-OOV bonus, improving outputs such as `is this a snorql` while preserving identifier-like `openai`.
- One- and two-character entries found only in bare spelling dictionaries are treated as weak evidence so abbreviations and letter entries do not shred OOV tokens.
- Ranked fallback vocabulary duplicates now retain their earliest (best) rank.
- Unknown candidates may use the configured `max_word_length` even when the loaded dictionary contains only shorter words.
- CLI subcommand option parsing delegates options to the selected subcommand instead of treating them as root options.
- The optional DKSplit boundary backend uses a calibrated neural weight of `0.5`; unsupported Unicode or overlength runs fall back to the existing heuristic path without truncation.

## [0.1.0] - 2026-09-24

### Added

- Strict case encoding, reversible decoding, conversion, compatibility detection, and ambiguity-aware case guessing in the Python API.
- Low-resource word segmentation using common vocabulary, system dictionaries, optional frequency data, custom and blocked words, unknown-word scoring, punctuation boundaries, capitalization hints, numeric boundaries, and lossless source spans.
- Initial command-line word segmentation support.
