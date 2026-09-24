"""
Low-resource word segmentation for lowercase text with no spaces/punctuation.

Examples
--------
    python wordsegment.py thisisatestofwordsegmentation

    from wordsegment import WordSegmenter
    seg = WordSegmenter()
    print(seg.segment("thisisatestofwordsegmentation"))

No third-party dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

# A small core vocabulary helps substantially when the system dictionary
# provides no frequency information.
#
# Roughly ordered from more common to less common.
COMMON_WORDS = [
    "the",
    "of",
    "and",
    "to",
    "a",
    "in",
    "is",
    "it",
    "you",
    "that",
    "he",
    "was",
    "for",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "i",
    "at",
    "be",
    "this",
    "have",
    "from",
    "or",
    "one",
    "had",
    "by",
    "word",
    "but",
    "not",
    "what",
    "all",
    "were",
    "we",
    "when",
    "your",
    "can",
    "said",
    "there",
    "use",
    "an",
    "each",
    "which",
    "she",
    "do",
    "how",
    "their",
    "if",
    "will",
    "up",
    "other",
    "about",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "him",
    "into",
    "time",
    "has",
    "look",
    "two",
    "more",
    "write",
    "go",
    "see",
    "number",
    "no",
    "way",
    "could",
    "people",
    "my",
    "than",
    "first",
    "water",
    "been",
    "call",
    "who",
    "oil",
    "its",
    "now",
    "find",
    "long",
    "down",
    "day",
    "did",
    "get",
    "come",
    "made",
    "may",
    "part",
    "over",
    "new",
    "sound",
    "take",
    "only",
    "little",
    "work",
    "know",
    "place",
    "year",
    "live",
    "me",
    "back",
    "give",
    "most",
    "very",
    "after",
    "thing",
    "our",
    "just",
    "name",
    "good",
    "sentence",
    "man",
    "think",
    "say",
    "great",
    "where",
    "help",
    "through",
    "much",
    "before",
    "line",
    "right",
    "too",
    "mean",
    "old",
    "any",
    "same",
    "tell",
    "boy",
    "follow",
    "came",
    "want",
    "show",
    "also",
    "around",
    "form",
    "three",
    "small",
    "set",
    "put",
    "end",
    "does",
    "another",
    "well",
    "large",
    "must",
    "big",
    "even",
    "such",
    "because",
    "turn",
    "here",
    "why",
    "ask",
    "went",
    "men",
    "read",
    "need",
    "land",
    "different",
    "home",
    "us",
    "move",
    "try",
    "kind",
    "hand",
    "picture",
    "again",
    "change",
    "off",
    "play",
    "spell",
    "air",
    "away",
    "animal",
    "house",
    "point",
    "page",
    "letter",
    "mother",
    "answer",
    "found",
    "study",
    "still",
    "learn",
    "should",
    "america",
    "world",
    "high",
    "every",
    "near",
    "add",
    "food",
    "between",
    "own",
    "below",
    "country",
    "plant",
    "last",
    "school",
    "father",
    "keep",
    "tree",
    "never",
    "start",
    "city",
    "earth",
    "eye",
    "light",
    "thought",
    "head",
    "under",
    "story",
    "saw",
    "left",
    "dont",
    "few",
    "while",
    "along",
    "might",
    "close",
    "something",
    "seem",
    "next",
    "hard",
    "open",
    "example",
    "begin",
    "life",
    "always",
    "those",
    "both",
    "paper",
    "together",
    "got",
    "group",
    "often",
    "run",
    "important",
    "until",
    "children",
    "side",
    "feet",
    "car",
    "mile",
    "night",
    "walk",
    "white",
    "sea",
    "began",
    "grow",
    "took",
    "river",
    "four",
    "carry",
    "state",
    "once",
    "book",
    "hear",
    "stop",
    "without",
    "second",
    "later",
    "miss",
    "idea",
    "enough",
    "eat",
    "face",
    "watch",
    "far",
    "indian",
    "really",
    "almost",
    "let",
    "above",
    "girl",
    "sometimes",
    "mountain",
    "cut",
    "young",
    "talk",
    "soon",
    "list",
    "song",
    "being",
    "leave",
    "family",
    "body",
    "music",
    "color",
    "stand",
    "sun",
    "questions",
    "fish",
    "area",
    "mark",
    "dog",
    "horse",
    "birds",
    "problem",
    "complete",
    "room",
    "knew",
    "since",
    "ever",
    "piece",
    "told",
    "usually",
    "didnt",
    "friends",
    "easy",
    "heard",
    "order",
    "red",
    "door",
    "sure",
    "become",
    "top",
    "ship",
    "across",
    "today",
    "during",
    "short",
    "better",
    "best",
    "however",
    "low",
    "hours",
    "black",
    "products",
    "happened",
    "whole",
    "measure",
    "remember",
    "early",
    "waves",
    "reached",
    "listen",
    "wind",
    "rock",
    "space",
    "covered",
    "fast",
    "several",
    "hold",
    "himself",
    "toward",
    "five",
    "step",
    "morning",
    "passed",
    "vowel",
    "true",
    "hundred",
    "against",
    "pattern",
    "numeral",
    "table",
    "north",
    "slowly",
    "money",
    "map",
    "farm",
    "pulled",
    "draw",
    "voice",
    "seen",
    "cold",
    "cried",
    "plan",
    "notice",
    "south",
    "sing",
    "war",
    "ground",
    "fall",
    "king",
    "town",
    "ill",
    "unit",
    "figure",
    "certain",
    "field",
    "travel",
    "wood",
    "fire",
    "upon",
    "done",
    "english",
    "road",
    "half",
    "ten",
    "fly",
    "gave",
    "box",
    "finally",
    "wait",
    "correct",
    "oh",
    "quickly",
    "person",
    "became",
    "shown",
    "minutes",
    "strong",
    "verb",
    "stars",
    "front",
    "feel",
    "fact",
    "inches",
    "street",
    "decided",
    "contain",
    "course",
    "surface",
    "produce",
    "building",
    "ocean",
    "class",
    "note",
    "nothing",
    "rest",
    "carefully",
    "scientists",
    "inside",
    "wheels",
    "stay",
    "green",
    "known",
    "island",
    "week",
    "less",
    "machine",
    "base",
    "ago",
    "stood",
    "plane",
    "system",
    "behind",
    "ran",
    "round",
    "boat",
    "game",
    "force",
    "brought",
    "understand",
    "warm",
    "common",
    "bring",
    "explain",
    "dry",
    "though",
    "language",
    "shape",
    "deep",
    "thousands",
    "yes",
    "clear",
    "equation",
    "yet",
    "government",
    "filled",
    "heat",
    "full",
    "hot",
    "check",
    "object",
    "am",
    "rule",
    "among",
    "noun",
    "power",
    "cannot",
    "able",
    "six",
    "size",
    "dark",
    "ball",
    "material",
    "special",
    "heavy",
    "fine",
    "pair",
    "circle",
    "include",
    "built",
    "cant",
    "matter",
    "square",
    "syllables",
    "perhaps",
    "bill",
    "felt",
    "suddenly",
    "test",
    "direction",
    "center",
    "farmers",
    "ready",
    "anything",
    "divided",
    "general",
    "energy",
    "subject",
    "europe",
    "moon",
    "region",
    "return",
    "believe",
    "dance",
    "members",
    "picked",
    "simple",
    "cells",
    "paint",
    "mind",
    "love",
    "cause",
    "rain",
    "exercise",
    "eggs",
    "train",
    "blue",
    "wish",
    "drop",
    "developed",
    "window",
    "difference",
    "distance",
    "heart",
    "sit",
    "sum",
    "summer",
    "wall",
    "forest",
    "probably",
    "legs",
    "sat",
    "main",
    "winter",
    "wide",
    "written",
    "length",
    "reason",
    "kept",
    "interest",
    "arms",
    "brother",
    "race",
    "present",
    "beautiful",
    "store",
    "job",
    "edge",
    "past",
    "sign",
    "record",
    "finished",
    "discovered",
    "wild",
    "happy",
    "beside",
    "gone",
    "sky",
    "glass",
    "million",
    "west",
    "lay",
    "weather",
    "root",
    "instruments",
    "meet",
    "third",
    "months",
    "paragraph",
    "raised",
    "represent",
    "soft",
    "whether",
    "clothes",
    "flowers",
    "shall",
    "teacher",
    "held",
    "describe",
    "drive",
    "cross",
    "speak",
    "solve",
    "appear",
    "metal",
    "son",
    "either",
    "ice",
    "sleep",
    "village",
    "factors",
    "result",
    "jumped",
    "snow",
    "ride",
    "care",
    "floor",
    "hill",
    "pushed",
    "baby",
    "buy",
    "century",
    "outside",
    "everything",
    "tall",
    "already",
    "instead",
    "phrase",
    "soil",
    "bed",
    "copy",
    "free",
    "hope",
    "spring",
    "case",
    "laughed",
    "nation",
    "quite",
    "type",
    "themselves",
    "temperature",
    "bright",
    "lead",
    "everyone",
    "method",
    "section",
    "lake",
    "iron",
    "within",
    "dictionary",
    "hair",
    "age",
    "amount",
    "scale",
    "pounds",
    "although",
    "per",
    "broken",
    "moment",
    "tiny",
    "possible",
    "gold",
    "milk",
    "quiet",
    "natural",
    "lot",
    "stone",
    "act",
    "build",
    "middle",
    "speed",
    "count",
    "consonant",
    "someone",
    "sail",
    "rolled",
    "bear",
    "wonder",
    "smiled",
    "angle",
    "fraction",
    "africa",
    "killed",
    "melody",
    "bottom",
    "trip",
    "hole",
    "poor",
    "lets",
    "fight",
    "surprise",
    "french",
    "died",
    "beat",
    "exactly",
    "remain",
    "dress",
    "cat",
    "couldnt",
    "fingers",
    "row",
    "least",
    "catch",
    "climbed",
    "wrote",
    "shouted",
    "continued",
    "itself",
    "else",
    "plains",
    "gas",
    "england",
    "burning",
    "design",
    "joined",
    "foot",
    "law",
    "ears",
    "grass",
    "youre",
    "grew",
    "skin",
    "valley",
    "cents",
    "key",
    "president",
    "brown",
    "trouble",
    "cool",
    "cloud",
    "lost",
    "sent",
    "symbols",
    "wear",
    "bad",
    "save",
    "experiment",
    "engine",
    "alone",
    "drawing",
    "east",
    "choose",
    "single",
    "touch",
    "information",
    "express",
    "mouth",
    "yard",
    "equal",
    "decimal",
    "yourself",
    "control",
    "practice",
    "report",
    "straight",
    "rise",
    "statement",
    "stick",
    "party",
    "seeds",
    "suppose",
    "woman",
    "coast",
    "bank",
    "period",
    "wire",
    "pay",
    "clean",
    "visit",
    "bit",
    "whose",
    "received",
    "garden",
    "please",
    "strange",
    "caught",
    "fell",
    "team",
    "god",
    "captain",
    "direct",
    "ring",
    "serve",
    "child",
    "desert",
    "increase",
    "history",
    "cost",
    "maybe",
    "business",
    "separate",
    "break",
    "uncle",
    "hunting",
    "flow",
    "lady",
    "students",
    "human",
    "art",
    "feeling",
    "supply",
    "corner",
    "electric",
    "insects",
    "crops",
    "tone",
    "hit",
    "sand",
    "doctor",
    "provide",
    "thus",
    "wont",
    "cook",
    "bones",
    "tail",
    "board",
    "modern",
    "compound",
    "mine",
    "wasnt",
    "fit",
    "addition",
    "belong",
    "safe",
    "soldiers",
    "guess",
    "silent",
    "trade",
    "rather",
    "compare",
    "crowd",
    "poem",
    "enjoy",
    "elements",
    "indicate",
    "except",
    "expect",
    "flat",
    "seven",
    "interesting",
    "sense",
    "string",
    "blow",
    "famous",
    "value",
    "wings",
    "movement",
    "pole",
    "exciting",
    "branches",
    "thick",
    "blood",
    "lie",
    "spot",
    "bell",
    "fun",
    "loud",
    "consider",
    "suggested",
    "thin",
    "position",
    "entered",
    "fruit",
    "tied",
    "rich",
    "dollars",
    "send",
    "sight",
    "chief",
    "japanese",
    "stream",
    "planets",
    "rhythm",
    "eight",
    "science",
    "major",
    "observe",
    "tube",
    "necessary",
    "weight",
    "meat",
    "lifted",
    "process",
    "army",
    "hat",
    "property",
    "particular",
    "swim",
    "terms",
    "current",
    "park",
    "sell",
    "shoulder",
    "industry",
    "wash",
    "block",
    "spread",
    "cattle",
    "wife",
    "sharp",
    "company",
    "radio",
    "well",
    "discuss",
    "actually",
    "real",
    "possible",
    "maybe",
    "also",
    "just",
    "even",
    "still",
    "already",
    "really",
    "probably",
    "perhaps",
]


SYSTEM_WORDLISTS = (
    # Linux / Unix
    "/usr/share/dict/words",
    "/usr/dict/words",
    # Some distributions/packages
    "/usr/share/dict/american-english",
    "/usr/share/dict/british-english",
    "/usr/share/hunspell/en_US.dic",
    "/usr/share/hunspell/en_GB.dic",
    "/usr/share/myspell/dicts/en_US.dic",
    # macOS
    "/usr/share/dict/words",
)


def normalize_word(word: str) -> str | None:
    """
    Normalize a dictionary entry.

    Keeps alphabetic words and internal apostrophes, then strips apostrophes
    because the input is assumed to have punctuation removed.

    Examples
    --------
        "don't" -> "dont"
        "John"  -> "john"
    """
    word = word.strip().lower()

    # Hunspell dictionaries may contain flags after "/".
    if "/" in word:
        word = word.split("/", 1)[0]

    # Remove common punctuation that may have vanished from input.
    word = word.replace("'", "").replace("\u2019", "")

    if not word or not word.isalpha():
        return None

    return word


class WordSegmenter:
    def __init__(
        self,
        wordlist: str | os.PathLike | None = None,
        frequency_file: str | os.PathLike | None = None,
        max_word_length: int = 32,
        unknown_base_cost: float = 12.0,
        unknown_char_cost: float = 1.8,
    ):
        """
        Create a word segmenter.

        wordlist:
            Optional dictionary file. If omitted, common system wordlists
            are searched automatically.

        frequency_file:
            Optional file with word frequencies. Supported formats include:

                the 23135851162
                of 13151942776
                ...

            or simply a frequency-ordered list:

                the
                of
                and
                ...

        max_word_length:
            Maximum dictionary word length considered.

        unknown_base_cost / unknown_char_cost:
            Penalty for unknown spans. Larger values make the algorithm
            prefer known dictionary words more strongly.
        """
        self.max_word_length = max_word_length
        self.unknown_base_cost = unknown_base_cost
        self.unknown_char_cost = unknown_char_cost

        self.words: set[str] = set()
        self.cost: dict[str, float] = {}

        self._load_common_words()

        if frequency_file:
            self._load_frequency_file(Path(frequency_file))

        if wordlist:
            self._load_dictionary(Path(wordlist))
        else:
            self._load_system_dictionary()

        # Avoid trying substrings longer than any known word.
        if self.words:
            self.max_word_length = min(
                self.max_word_length,
                max(map(len, self.words)),
            )

    def _load_common_words(self) -> None:
        """
        Give common words rank-based costs.

        Lower cost = more probable.
        """
        n = len(COMMON_WORDS)

        for rank, word in enumerate(COMMON_WORDS, 1):
            self.words.add(word)

            # Approximate Zipf-like rank probability.
            #
            # The constant isn't important; relative costs are.
            probability = 1.0 / (rank * math.log(n + 1))
            self.cost[word] = -math.log(probability)

    def _add_dictionary_lines(self, lines: Iterable[str]) -> None:
        first = True
        for line in lines:
            if first:
                first = False
                if line.strip().isdigit():
                    continue

            word = normalize_word(line)
            if word and len(word) <= self.max_word_length:
                self.words.add(word)

    def _load_dictionary(self, path: Path) -> bool:
        if not path.is_file():
            return False

        try:
            handle = path.open("r", encoding="utf-8", errors="ignore")
        except OSError:
            return False

        with handle:
            self._add_dictionary_lines(handle)
        return True

    def _load_system_dictionary(self) -> None:
        for filename in SYSTEM_WORDLISTS:
            path = Path(filename)
            if path.is_file() and self._load_dictionary(path):
                # One good system dictionary is enough.
                return

    @staticmethod
    def _parse_frequency(line: str) -> tuple[str, float | None] | None:
        pieces = line.strip().split()
        if not pieces:
            return None

        word = normalize_word(pieces[0])
        if not word:
            return None

        frequency = None
        if len(pieces) > 1:
            try:
                frequency = float(pieces[1])
            except ValueError:
                pass

        return word, frequency

    def _read_frequency_entries(self, path: Path) -> list[tuple[str, float | None]] | None:
        try:
            handle = path.open("r", encoding="utf-8", errors="ignore")
        except OSError:
            return None

        with handle:
            return [entry for line in handle if (entry := self._parse_frequency(line)) is not None]

    def _load_weighted_frequencies(self, entries: list[tuple[str, float | None]]) -> bool:
        total = sum(freq for _, freq in entries if freq is not None and freq > 0)
        if total <= 0:
            return False

        for word, freq in entries:
            self.words.add(word)
            if freq is not None and freq > 0:
                self.cost[word] = -math.log(freq / total)
        return True

    def _load_ranked_frequencies(self, entries: list[tuple[str, float | None]]) -> None:
        harmonic_normalizer = math.log(len(entries) + 1)
        for rank, (word, _) in enumerate(entries, 1):
            self.words.add(word)
            probability = 1.0 / (rank * harmonic_normalizer)
            self.cost[word] = -math.log(probability)

    def _load_frequency_file(self, path: Path) -> bool:
        """Load frequencies or a list ordered by decreasing frequency."""
        if not path.is_file():
            return False

        entries = self._read_frequency_entries(path)
        if not entries:
            return False

        if any(freq is not None for _, freq in entries):
            return self._load_weighted_frequencies(entries)

        self._load_ranked_frequencies(entries)
        return True

    def word_cost(self, word: str) -> float:
        """
        Return the cost of a candidate word.

        Known frequent words are cheap.
        Dictionary-only words use a heuristic.
        Unknown words are expensive but possible.
        """
        explicit = self.cost.get(word)
        if explicit is not None:
            return explicit

        if word in self.words:
            # Dictionary-only word.
            #
            # Slightly prefer longer dictionary matches because otherwise
            # arbitrary combinations of many tiny words can win.
            #
            # Examples:
            #   "there" should usually beat "the re"
            #   "another" should usually beat "an other"
            return 10.5 - min(len(word), 12) * 0.30

        # Unknown word.
        #
        # The first characters are expensive, then extending an already
        # unknown token becomes comparatively cheaper. This keeps things
        # like names or technical terms together instead of splitting every
        # character separately.
        return self.unknown_base_cost + self.unknown_char_cost * math.sqrt(len(word))

    def _segment_letters(self, text: str) -> list[str]:
        """Return the best segmentation for a lowercase alphabetic run."""
        n = len(text)

        # dp[i] = cheapest cost for text[:i]
        dp = [math.inf] * (n + 1)

        # back[i] = starting position of final token in best text[:i]
        back = [-1] * (n + 1)

        dp[0] = 0.0

        for end in range(1, n + 1):
            start_min = max(0, end - self.max_word_length)

            for start in range(start_min, end):
                word = text[start:end]

                candidate = dp[start] + self.word_cost(word)

                if candidate < dp[end]:
                    dp[end] = candidate
                    back[end] = start

        result = []
        pos = n

        while pos > 0:
            start = back[pos]

            if start < 0:
                # Shouldn't normally happen because unknown words are allowed.
                return [text]

            result.append(text[start:pos])
            pos = start

        result.reverse()
        return result

    def segment(self, text: str) -> list[str]:
        """Return the best segmentation while preserving numeric runs."""
        text = "".join(c.lower() for c in text if c.isalnum())

        if not text:
            return []

        result: list[str] = []
        start = 0

        for index in range(1, len(text) + 1):
            at_end = index == len(text)
            changes_kind = (
                not at_end
                and text[index - 1].isdigit() != text[index].isdigit()
            )

            if not at_end and not changes_kind:
                continue

            run = text[start:index]
            if run.isdigit():
                result.append(run)
            else:
                result.extend(self._segment_letters(run))
            start = index

        return result

    def segment_string(self, text: str) -> str:
        return " ".join(self.segment(text))

