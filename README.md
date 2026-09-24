# wordrobe

> A costume change for words.

`wordrobe` is a small Python utility for recognizing, decoding, converting, and heuristically recovering words from common identifier conventions such as `snake_case`, `SCREAMING_SNAKE_CASE`, `kebab-case`, `camelCase`, `PascalCase`, and delimiter-free text.

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

Delimiter-based cases are decoded strictly and losslessly by the Python `decode()` API. Cases such as `camelCase`, `PascalCase`, and `flatcase` cannot be inverted exactly because capitalization does not preserve every possible word boundary.

`possible_cases()` reports every syntactically compatible case in deterministic preference order. `guess_case()` answers the separate question of which case convention is identifiable from syntax; it is not required for heuristic word recovery.

## Word recovery

`WordSegmenter` recovers words from unseparated or mixed-format text using minimum-cost dynamic programming over lexical and boundary evidence. It uses a compact built-in ranked vocabulary, an available system dictionary when present, optional frequency information, capitalization and numeric transitions, and configurable unknown-word costs.

```python
from wordrobe.segment import WordSegmenter

segmenter = WordSegmenter()

segmenter.segment("thisisatest")
# ['this', 'is', 'a', 'test']

segmenter.segment("isthisacamel")
# ['is', 'this', 'a', 'camel']

segmenter.segment("parseHTTPResponseBody")
# ['parse', 'HTTP', 'Response', 'Body']
```

Punctuation is treated as a hard boundary. `segment()` returns only tokens; `segment_spans()` provides a lossless representation including separators and source offsets:

```python
spans = segmenter.segment_spans("load-xstate_statechart")
"".join(span.text for span in spans)
# 'load-xstate_statechart'
```

### Scoring

Known words receive frequency/rank-derived costs. Unknown candidates use a base penalty plus a **linear per-character cost**, so concatenating several words into one long unknown token does not become artificially cheap. Very short unknown fragments receive a small additional penalty to reduce artifacts such as splitting an unknown word into a known prefix plus a stray letter.

`unknown_base_cost` and `unknown_char_cost` tune the default unknown model. A custom callback can replace it entirely:

```python
def unknown_cost(word: str) -> float:
    return 2.0 if word.startswith("x") else 20.0

segmenter = WordSegmenter(unknown_cost=unknown_cost)
```

The callback must return a finite numeric cost.

### Custom vocabulary

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

### Dictionaries, frequencies, and boundary hints

`wordlist=` selects an explicit dictionary file. If omitted, `WordSegmenter` searches common system wordlist locations. `frequency_file=` accepts either `word frequency` lines or a list ordered from most to least frequent.

Case transitions such as `parseHTTPResponse` and transitions between digits and letters provide soft boundary bonuses. Set `use_case_hints=False` to disable capitalization evidence. `case_boundary_bonus` and `numeric_boundary_bonus` control the strength of these hints.

## CLI

`decode` is the canonical word-recovery command. When `--case` is omitted it does **not** first require a unique case classification; it directly combines explicit boundaries, case transitions, numeric transitions, and lexical segmentation.

```bash
wordrobe encode --case camelCase hello world
# helloWorld

wordrobe decode hello_world
# hello world

wordrobe decode isThisACamel
# is this a camel

wordrobe decode isthisacamel
# is this a camel

wordrobe decode --case snake_case hello_world
# hello world

wordrobe convert --to snake_case isThisACamel
# is_this_a_camel

wordrobe convert --from snake_case --to PascalCase hello_world
# HelloWorld

wordrobe guess hello_world
# snake_case

wordrobe guess --all hello
# prints every compatible case in deterministic preference order
```

An explicit `--case` asks for that syntax to be validated. Reversible cases are then decoded exactly; implicit-boundary cases still require heuristic segmentation after validation.

The former `segment` command remains as a hidden compatibility alias for `decode`; there is no longer a separate CLI segmentation policy. `guess` remains separate because identifying a case convention is a different question from recovering probable word boundaries.
