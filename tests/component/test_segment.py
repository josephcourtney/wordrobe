import math

import pytest

from wordrobe.segment import COMMON_WORDS, SegmentSpan, WordSegmenter


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


def test_viterbi_splits_unseparated_common_words_without_system_dictionary(tmp_path) -> None:
    segmenter = WordSegmenter(wordlist=tmp_path / "missing-dictionary")

    assert segmenter.segment("thisisatest") == ["this", "is", "a", "test"]
    assert segmenter.segment("isthisacamel") == ["is", "this", "a", "camel"]


def test_oov_cohesion_prefers_article_plus_unknown_noun(tmp_path) -> None:
    dictionary = tmp_path / "words"
    dictionary.write_text("nor\nq\nl\n", encoding="utf-8")
    segmenter = WordSegmenter(wordlist=dictionary)

    assert segmenter.segment("isthisasnorql") == ["is", "this", "a", "snorql"]
    assert segmenter.segment("thisisaxyzzy") == ["this", "is", "a", "xyzzy"]


def test_oov_cohesion_keeps_identifier_like_singleton_sequences_together(tmp_path) -> None:
    dictionary = tmp_path / "words"
    dictionary.write_text("ai\nq\nl\n", encoding="utf-8")
    segmenter = WordSegmenter(wordlist=dictionary)

    assert segmenter.segment("openai") == ["openai"]
    assert segmenter.segment("scroot") == ["scroot"]


def test_oov_cohesion_still_extracts_strong_known_prefixes(tmp_path) -> None:
    segmenter = WordSegmenter(wordlist=tmp_path / "missing-dictionary")

    assert segmenter.segment("loadfrobnicate") == ["load", "frobnicate"]


def test_dictionary_only_short_entries_are_weak_evidence(tmp_path) -> None:
    dictionary = tmp_path / "words"
    dictionary.write_text("q\nai\nnor\n", encoding="utf-8")
    segmenter = WordSegmenter(wordlist=dictionary)

    assert segmenter.word_cost("q") > segmenter.word_cost("nor")
    assert segmenter.word_cost("ai") > segmenter.word_cost("nor")


def test_default_unknown_cost_grows_linearly_with_length(tmp_path) -> None:
    segmenter = WordSegmenter(wordlist=tmp_path / "missing-dictionary")

    assert segmenter.word_cost("qzxvbnm") == pytest.approx(12.0 + 4.0 * 7)
    assert segmenter.word_cost("q") == pytest.approx(12.0 + 4.0 + 20.0)


def test_common_short_word_can_split_from_unknown_remainder(tmp_path) -> None:
    segmenter = WordSegmenter(wordlist=tmp_path / "missing-dictionary")

    assert segmenter.segment("aqzxvbnm") == ["a", "qzxvbnm"]


def test_short_unknown_fragment_penalty_avoids_known_suffix_oversegmentation(tmp_path) -> None:
    segmenter = WordSegmenter(wordlist=tmp_path / "missing-dictionary")

    assert segmenter.segment("scroot") == ["scroot"]


def test_long_unknown_candidate_is_not_limited_by_dictionary_word_length(tmp_path) -> None:
    target = "qzxvbnmqzxvbnmqzxvbnm"

    def unknown_cost(word: str) -> float:
        return 0.0 if word == target else 100.0

    segmenter = WordSegmenter(
        wordlist=tmp_path / "missing-dictionary",
        max_word_length=32,
        unknown_cost=unknown_cost,
    )

    assert segmenter.segment(target) == [target]


def test_duplicate_common_words_keep_their_best_rank(tmp_path) -> None:
    segmenter = WordSegmenter(wordlist=tmp_path / "missing-dictionary")
    word = "just"
    first_rank = COMMON_WORDS.index(word) + 1
    expected = math.log(first_rank * math.log(len(COMMON_WORDS) + 1))

    assert COMMON_WORDS.count(word) > 1
    assert segmenter.cost[word] == pytest.approx(expected)


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


def test_empty_input_has_no_segments_or_spans() -> None:
    segmenter = WordSegmenter()

    assert segmenter.segment("") == []
    assert segmenter.segment_spans("") == []
    assert segmenter.segment_string("") == ""


@pytest.mark.parametrize("cost", [float("inf"), float("-inf"), float("nan")])
def test_custom_word_costs_must_be_finite(cost: float) -> None:
    with pytest.raises(ValueError, match="custom word cost must be finite"):
        WordSegmenter(extra_words={"scroot": cost})


def test_unknown_word_callback_cost_must_be_finite() -> None:
    segmenter = WordSegmenter(unknown_cost=lambda _word: float("nan"))

    with pytest.raises(ValueError, match="unknown-word cost must be finite"):
        segmenter.segment("xyzzy")


@pytest.mark.parametrize(
    "option",
    ["weak_short_word_penalty", "adjacent_singleton_penalty", "article_unknown_bonus"],
)
def test_sequence_scoring_options_must_be_finite_and_nonnegative(option: str) -> None:
    with pytest.raises(ValueError, match=option):
        WordSegmenter(**{option: float("nan")})
    with pytest.raises(ValueError, match=option):
        WordSegmenter(**{option: -1.0})


def test_segment_spans_preserves_consecutive_punctuation_as_one_separator() -> None:
    segmenter = WordSegmenter(extra_words={"foo": 1.0, "bar": 1.0})

    spans = segmenter.segment_spans("foo--bar")

    assert spans == [
        SegmentSpan("foo", 0, 3, "token"),
        SegmentSpan("--", 3, 5, "separator"),
        SegmentSpan("bar", 5, 8, "token"),
    ]
    assert "".join(span.text for span in spans) == "foo--bar"
