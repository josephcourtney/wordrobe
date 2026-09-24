# wordrobe

> A costume change for words.

`wordrobe` is a small Python utility for recognizing, decoding, and converting common word-casing conventions such as `snake_case`, `SCREAMING_SNAKE_CASE`, `kebab-case`, `camelCase`, and `PascalCase`.

## Development setup

From a checkout of the repository:

```bash
uv sync
```

## Usage

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

## Reversibility

Delimiter-based cases are decoded strictly and losslessly. Cases such as `camelCase`, `PascalCase`, and `flatcase` can be encoded, but are not decoded by guessing because capitalization does not preserve every possible word boundary.

`possible_cases()` reports every compatible case in deterministic preference order. `guess_case()` returns a case only when the interpretation is unique; with `should_raise=True`, it distinguishes invalid input from ambiguous input.

## Segmentation

`WordSegmenter` can heuristically recover words from unseparated text. Numeric runs are preserved as their own tokens:

```python
from wordrobe.segment import WordSegmenter

WordSegmenter().segment("word2number")
# ['word', '2', 'number']
```

## CLI

```bash
wordrobe --help
wordrobe segment thisisatest
```
