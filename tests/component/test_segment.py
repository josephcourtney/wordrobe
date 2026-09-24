from wordrobe.segment import SegmentSpan, WordSegmenter


def test_segment_preserves_numeric_runs() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("word2number") == ["word", "2", "number"]


def test_segment_preserves_consecutive_digits_as_one_token() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("word123number") == ["word", "123", "number"]


def test_segment_uses_punctuation_as_hard_boundaries() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("word-2-number") == ["word", "2", "number"]


def test_segment_string_joins_tokens() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment_string("word2number") == "word 2 number"


def test_segment_preserves_all_numeric_input() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("123") == ["123"]


def test_custom_words_accept_explicit_costs() -> None:
    segmenter = WordSegmenter(extra_words={"scroot": 1.0, "status": 1.0})

    assert segmenter.segment("scrootstatus") == ["scroot", "status"]


def test_custom_words_accept_a_collection_and_default_cost() -> None:
    segmenter = WordSegmenter(extra_words={"frobnicate", "widget"}, custom_word_cost=1.0)

    assert segmenter.segment("frobnicatewidget") == ["frobnicate", "widget"]


def test_blocked_words_force_an_alternative_segmentation() -> None:
    segmenter = WordSegmenter(
        extra_words={"foobar": 1.0, "foo": 2.0, "bar": 2.0},
        blocked_words={"foobar"},
    )

    assert segmenter.segment("foobar") == ["foo", "bar"]


def test_unknown_cost_callback_controls_unknown_word_scoring() -> None:
    def unknown_cost(word: str) -> float:
        return 0.0 if word == "xyzzy" else 100.0

    segmenter = WordSegmenter(extra_words={"load": 1.0}, unknown_cost=unknown_cost)

    assert segmenter.segment("xyzzyload") == ["xyzzy", "load"]


def test_camel_case_boundaries_are_used_and_case_is_preserved() -> None:
    segmenter = WordSegmenter(extra_words={"parse": 1.0, "http": 1.0, "response": 1.0, "body": 1.0})

    assert segmenter.segment("parseHTTPResponseBody") == ["parse", "HTTP", "Response", "Body"]


def test_acronym_to_capitalized_word_boundary_is_used() -> None:
    segmenter = WordSegmenter(extra_words={"http": 1.0, "server": 1.0})

    assert segmenter.segment("HTTPServer") == ["HTTP", "Server"]


def test_custom_word_can_span_an_internal_case_hint() -> None:
    segmenter = WordSegmenter(extra_words={"xstate": 1.0, "loader": 1.0})

    assert segmenter.segment("XStateLoader") == ["XState", "Loader"]


def test_kebab_and_snake_case_create_hard_boundaries() -> None:
    segmenter = WordSegmenter(extra_words={"load": 1.0, "xstate": 1.0, "statechart": 1.0})

    assert segmenter.segment("load-xstate_statechart") == ["load", "xstate", "statechart"]


def test_alphanumeric_custom_words_can_span_letter_digit_boundaries() -> None:
    segmenter = WordSegmenter(extra_words={"gpt4": 1.0, "model": 1.0})

    assert segmenter.segment("gpt4model") == ["gpt4", "model"]


def test_segment_spans_preserves_separators_and_offsets() -> None:
    segmenter = WordSegmenter(extra_words={"load": 1.0, "xstate": 1.0, "statechart": 1.0})

    spans = segmenter.segment_spans("load-xstate_statechart")

    assert spans == [
        SegmentSpan("load", 0, 4, "token"),
        SegmentSpan("-", 4, 5, "separator"),
        SegmentSpan("xstate", 5, 11, "token"),
        SegmentSpan("_", 11, 12, "separator"),
        SegmentSpan("statechart", 12, 22, "token"),
    ]
    assert "".join(span.text for span in spans) == "load-xstate_statechart"
