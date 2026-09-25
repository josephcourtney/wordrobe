# wordrobe

> A costume change for words.

`wordrobe` is a small Python utility for splitting, identifying, and converting common identifier conventions such as `snake_case`, `SCREAMING_SNAKE_CASE`, `kebab-case`, `camelCase`, `PascalCase`, and delimiter-free text.

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

`WordSegmenter` recovers words from unseparated or mixed-format text using minimum-cost dynamic programming over lexical, boundary, and small amounts of local sequence evidence. It uses a compact built-in ranked vocabulary, an available system dictionary when present, optional frequency information, capitalization and numeric transitions, and configurable unknown-word costs.

```python
from wordrobe.segment import WordSegmenter

segmenter = WordSegmenter()

segmenter.segment("thisisatest")
# ['this', 'is', 'a', 'test']

segmenter.segment("isthisacamel")
# ['is', 'this', 'a', 'camel']

segmenter.segment("isthisasnorql")
# ['is', 'this', 'a', 'snorql']

segmenter.segment("openai")
# ['openai']

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

Known words receive frequency/rank-derived costs. Unknown candidates use a base penalty plus a **linear per-character cost**, so concatenating several words into one long unknown token does not become artificially cheap. One- and two-character unknown fragments receive an additional penalty so the model can extract common short words without routinely breaking technical identifiers around known prefixes or suffixes.

Every boundary inferred inside one uninterrupted alphanumeric run also receives a modest cohesion penalty. Explicit punctuation creates a hard boundary and does not pay this cost; capitalization and letter/digit transitions provide positive boundary evidence that offsets it. This prevents weak lexical coincidences from creating too many cuts while still allowing strong known words to separate from OOV text.

The Viterbi state carries two small pieces of local context that a scalar token cost cannot express:

- consecutive implicit one-letter words are penalized, which keeps identifier-like forms such as `openai` together instead of producing `open a i`;
- an article (`a`, `an`, `the`) that occurs after already recovered context receives a modest bonus before an unknown word, allowing `is this a snorql` to beat `is this as norql` without encouraging a completely unknown identifier to be split merely because it starts with an article string.

Bare spelling dictionaries are treated as weaker evidence than ranked or frequency-backed vocabulary. In particular, one- and two-character dictionary-only entries receive a penalty because system dictionaries often contain abbreviations, letters, or symbols such as `q`, `l`, or `ai` that would otherwise fragment an OOV token.

`unknown_base_cost` and `unknown_char_cost` tune the default unknown model. `weak_short_word_penalty`, `implicit_boundary_penalty`, `adjacent_singleton_penalty`, and `article_unknown_bonus` tune the cohesion and local sequence heuristics. A custom callback can replace unknown-word scoring entirely:

```python
def unknown_cost(word: str) -> float:
    return 2.0 if word.startswith("x") else 20.0

segmenter = WordSegmenter(unknown_cost=unknown_cost)
```

The callback must return a finite numeric cost. The sequence-scoring penalties and bonus must be finite and non-negative.

### Optional neural boundary evidence

The default backend remains the dependency-light heuristic model. Installing the `neural` extra enables an optional DKSplit-derived character boundary model implemented by Wordrobe's stripped NumPy runtime:

```bash
pip install 'wordrobe[neural]'
```

From a repository checkout, use `uv sync --extra neural` instead. Then select the backend explicitly:

```python
from wordrobe.segment import BoundaryModel, WordSegmenter

segmenter = WordSegmenter(boundary_model=BoundaryModel.DKSPLIT)
segmenter.segment("isthisacamel")
# ['is', 'this', 'a', 'camel']
```

The neural model does not replace the lexical segmenter or restrict it to the neural model's preferred splits. Its exact CRF sequence score is added as another term in the same Viterbi search, so custom and blocked words, OOV costs, capitalization and numeric hints, and the existing local sequence rules remain active. `neural_weight` controls its strength; the default is `0.5`, calibrated to preserve the established adversarial regression suite.

The converted model supports ASCII alphanumeric runs up to 64 characters. Other runs fall back to the normal heuristic path instead of being truncated or mapped through an unknown-character token. Model provenance and attribution are recorded in `THIRD_PARTY_NOTICES.md` and the packaged `wordrobe/data/DKSPLIT_NOTICE.txt`.

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

The CLI is organized around four operations plus case discovery:

```text
split:    text  -> component words
join:     words -> formatted text
convert:  text  -> formatted text
case:     text  -> case classification
cases:            supported case conventions
```

Bare `wordrobe` prints the command overview, and `wordrobe --version` prints the installed version.

### Split words

`split` recovers semantic component words. With no `--from`, it directly combines explicit boundaries, case and numeric transitions, lexical evidence, and the selected boundary model. `--from` validates the input case; reversible cases are then split exactly.

```bash
wordrobe split hello_world
# hello world

wordrobe split isThisACamel
# is this a camel

wordrobe split isthisacamel
# is this a camel

wordrobe split isthisasnorql
# is this a snorql

wordrobe split isthisacamel --boundary-model dksplit
# is this a camel

wordrobe split hello_world --from snake_case
# hello world
```

The optional DKSplit backend remains explicit. Installing NumPy does not silently change segmentation behavior.

### Join words

`join` formats already-separated semantic words in a target convention:

```bash
wordrobe join hello world --to camelCase
# helloWorld

wordrobe join api client --to SCREAMING_SNAKE_CASE
# API_CLIENT
```

### Convert text

`convert` is the composition of splitting and joining. `--from` describes the input convention; `--to` always describes the output convention.

```bash
wordrobe convert isThisACamel --to snake_case
# is_this_a_camel

wordrobe convert hello_world --from snake_case --to PascalCase
# HelloWorld

wordrobe convert isthisasnorql --to kebab-case --boundary-model dksplit
# is-this-a-snorql
```

### Identify cases

`case` identifies a uniquely determined convention. `--all` prints every syntactically compatible convention in deterministic preference order.

```bash
wordrobe case hello_world
# snake_case

wordrobe case hello --all
# prints every compatible case in deterministic preference order
```

`cases` lists all supported conventions and indicates whether their word boundaries are explicit or must be inferred:

```bash
wordrobe cases
# snake_case               explicit boundaries
# ...
# camelCase                inferred boundaries
```

Case identification remains separate from word recovery: identifying a syntax and recovering probable semantic word boundaries are different questions.
