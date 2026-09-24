# wordrobe

> A costume change for words.

`wordrobe` is a small Python utility for recognizing, decoding, converting, and heuristically segmenting common word and identifier conventions such as `snake_case`, `SCREAMING_SNAKE_CASE`, `kebab-case`, `camelCase`, and `PascalCase`.

## Development setup

From a checkout of the repository:

```bash
uv sync
```

## Case API

```python
from wordrobe import Case, decode, encode, guess_case, possible_cases, translate

words = ["hello", "world"]

encode(words, Case.SNAKE)
# 'hello_world'

encode(words, Case.CAMEL)
# 'helloWorld'

decode("hello_world", Case.SNAKE)
# ['hello', 'world']

translate("hello_world", Case.SNAKE, Case.PASCAL)
# 'HelloWorld'

guess_case("hello_world")
# Case.SNAKE

possible_cases("hello")
# Multiple candidates: a single word does not reveal which case convention produced it.
```

Component words are normalized to lowercase before encoding and must be non-empty ASCII alphanumeric strings. This deliberately treats words as semantic components rather than preserving their original capitalization.

## Reversibility

Delimiter-based cases are decoded strictly and losslessly. Cases such as `camelCase`, `PascalCase`, and `flatcase` can be encoded, but are not decoded by the strict Python `decode()` API because capitalization does not preserve every possible word boundary.

`possible_cases()` reports every compatible case in deterministic preference order. `guess_case()` returns a case only when the interpretation is unique; with `should_raise=True`, it distinguishes invalid input from ambiguous input.

## Segmentation

`WordSegmenter` heuristically recovers words from unseparated or mixed-format text. It uses a small built-in common-word vocabulary, an available system dictionary when present, optional frequency information, and explicit boundary evidence. Numeric runs are preserved as tokens.

```python
from wordrobe.segment import WordSegmenter

segmenter = WordSegmenter()
segmenter.segment("word2number")
# ['word', '2', 'number']

segmenter.segment("parseHTTPResponseBody")
# Uses case transitions as soft boundary evidence while preserving original spelling.
```

Punctuation is treated as a hard boundary. `segment()` returns only tokens; `segment_spans()` provides a lossless representation including separators and source offsets:

```python
spans = segmenter.segment_spans("load-xstate_statechart")
"".join(span.text for span in spans)
# 'load-xstate_statechart'
```

### Custom vocabulary and scoring

Known project- or domain-specific words can be supplied as a collection using `custom_word_cost`, or as a mapping with explicit costs. Lower costs make a candidate more favorable.

```python
segmenter = WordSegmenter(
    extra_words={
        "scroot": 1.0,
        "statechart": 2.0,
    },
    blocked_words={"root"},
)

segmenter.segment("scrootstatechart")
# ['scroot', 'statechart']
```

`blocked_words` removes otherwise valid candidates from consideration. All explicit custom costs must be finite.

Unknown-word behavior can be customized with a callback:

```python
def unknown_cost(word: str) -> float:
    return 2.0 if word.startswith("x") else 20.0

segmenter = WordSegmenter(unknown_cost=unknown_cost)
```

The callback must return a finite numeric cost. Without one, unknown spans use the configurable `unknown_base_cost` and `unknown_char_cost` heuristic.

### Dictionaries, frequencies, and boundary hints

`wordlist=` selects an explicit dictionary file. If omitted, `WordSegmenter` searches common system wordlist locations. `frequency_file=` accepts either `word frequency` lines or a list ordered from most to least frequent.

Case transitions such as `parseHTTPResponse` and transitions between digits and letters provide soft boundary bonuses. Set `use_case_hints=False` to disable capitalization evidence. `case_boundary_bonus` and `numeric_boundary_bonus` control the strength of these hints.

## CLI

The command line exposes the same case semantics plus segmentation. `decode` infers a unique case when `--case` is omitted:

```bash
wordrobe encode --case camelCase hello world
# helloWorld

wordrobe decode hello_world
# hello world

wordrobe decode isThisACamel
# is this a camel

wordrobe decode --case snake_case hello_world
# hello world

wordrobe convert --from snake_case --to PascalCase hello_world
# HelloWorld

wordrobe guess hello_world
# snake_case

wordrobe guess --all hello
# prints every compatible case in deterministic preference order

wordrobe segment thisisatest
```

When `decode` infers or is explicitly given a reversible case, decoding remains strict and canonical. For implicit-boundary cases such as `camelCase`, `PascalCase`, `flatcase`, and `UPPERFLATCASE`, the CLI uses `WordSegmenter` to recover likely boundaries and emits lowercase semantic words. If the input is compatible with multiple cases, `decode` requires an explicit `--case` instead of choosing one arbitrarily.

`guess` exits with an error for ambiguous or invalid input unless `--all` is requested.
