from wordrobe.segment import WordSegmenter


def test_segment_preserves_numeric_runs() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("word2number") == ["word", "2", "number"]


def test_segment_preserves_consecutive_digits_as_one_token() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("word123number") == ["word", "123", "number"]


def test_segment_still_ignores_punctuation() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("word-2-number") == ["word", "2", "number"]


def test_segment_string_joins_tokens() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment_string("word2number") == "word 2 number"


def test_segment_preserves_all_numeric_input() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("123") == ["123"]
