# ruff: file-ignore: INP001
"""Minimal NumPy inference for DKSplit's fixed BiLSTM-CRF architecture.

This module is intentionally limited to code that could plausibly become a
wordrobe runtime backend. It depends only on NumPy and a converted weights NPZ.
ONNX inspection, conversion, and parity testing live under ``scripts/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"
UNK_IDX = 1
MAX_LEN = 64
HIDDEN_SIZE = 384
NUM_LAYERS = 3
NUM_DIRECTIONS = 2
NUM_TAGS = 2
FORMAT_VERSION = 1
VOCAB_SIZE = 38
_EMISSION_NDIM = 2
_UNIQUE_PATH_MULTIPLIER = 2

# ONNX Runtime MLAS logistic coefficients.
_LOGISTIC_ALPHA_9 = np.float32(4.37031012579801e-11)
_LOGISTIC_ALPHA_7 = np.float32(1.15627324459942e-07)
_LOGISTIC_ALPHA_5 = np.float32(6.08574864600143e-05)
_LOGISTIC_ALPHA_3 = np.float32(8.51377133304701e-03)
_LOGISTIC_ALPHA_1 = np.float32(2.48287947061529e-01)
_LOGISTIC_BETA_10 = np.float32(6.10247389755681e-13)
_LOGISTIC_BETA_8 = np.float32(5.76102136993427e-09)
_LOGISTIC_BETA_6 = np.float32(6.29106785017040e-06)
_LOGISTIC_BETA_4 = np.float32(1.70198817374094e-03)
_LOGISTIC_BETA_2 = np.float32(1.16817656904453e-01)
_LOGISTIC_BETA_0 = np.float32(9.93151921023180e-01)

# ONNX Runtime MLAS tanh coefficients.
_TANH_ALPHA_13 = np.float32(-2.76076847742355e-16)
_TANH_ALPHA_11 = np.float32(2.00018790482477e-13)
_TANH_ALPHA_9 = np.float32(-8.60467152213735e-11)
_TANH_ALPHA_7 = np.float32(5.12229709037114e-08)
_TANH_ALPHA_5 = np.float32(1.48572235717979e-05)
_TANH_ALPHA_3 = np.float32(6.37261928875436e-04)
_TANH_ALPHA_1 = np.float32(4.89352455891786e-03)
_TANH_BETA_6 = np.float32(1.19825839466702e-06)
_TANH_BETA_4 = np.float32(1.18534705686654e-04)
_TANH_BETA_2 = np.float32(2.26843463243900e-03)
_TANH_BETA_0 = np.float32(4.89352518554385e-03)

_CHAR_MAP = np.full(128, UNK_IDX, dtype=np.int64)
for _index, _char in enumerate(CHAR_VOCAB, start=2):
    _CHAR_MAP[ord(_char)] = _index


def text_to_ids(text: str) -> np.ndarray:
    """Apply DKSplit's lowercase, truncation, and ASCII character mapping."""
    processed = text.lower()[:MAX_LEN]
    raw = np.frombuffer(processed.encode("ascii", errors="replace"), dtype=np.uint8)
    return _CHAR_MAP[np.clip(raw, 0, 127)]


def _dequantize(q: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
    qf = np.asarray(q, dtype=np.float32)
    scale_f = np.asarray(scale, dtype=np.float32)
    zp_f = np.asarray(zero_point, dtype=np.float32)
    if scale_f.ndim == 0:
        return (qf - zp_f) * scale_f
    shape = (scale_f.shape[0],) + (1,) * (qf.ndim - 1)
    return (qf - zp_f.reshape(shape)) * scale_f.reshape(shape)


def _combined_bias(bias: np.ndarray) -> np.ndarray:
    expected = (NUM_DIRECTIONS, 8 * HIDDEN_SIZE)
    if bias.shape != expected:
        message = f"unexpected LSTM bias shape: {bias.shape}; expected {expected}"
        raise ValueError(message)
    return bias[:, : 4 * HIDDEN_SIZE] + bias[:, 4 * HIDDEN_SIZE :]


def _logistic(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float32), np.float32(-18.0), np.float32(18.0))
    x2 = x * x
    p = x2 * _LOGISTIC_ALPHA_9 + _LOGISTIC_ALPHA_7
    p = p * x2 + _LOGISTIC_ALPHA_5
    p = p * x2 + _LOGISTIC_ALPHA_3
    p = p * x2 + _LOGISTIC_ALPHA_1
    p *= x
    q = x2 * _LOGISTIC_BETA_10 + _LOGISTIC_BETA_8
    q = q * x2 + _LOGISTIC_BETA_6
    q = q * x2 + _LOGISTIC_BETA_4
    q = q * x2 + _LOGISTIC_BETA_2
    q = q * x2 + _LOGISTIC_BETA_0
    return np.clip(p / q + np.float32(0.5), np.float32(0.0), np.float32(1.0))


def _tanh(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float32), np.float32(-9.0), np.float32(9.0))
    x2 = x * x
    p = x2 * _TANH_ALPHA_13 + _TANH_ALPHA_11
    p = p * x2 + _TANH_ALPHA_9
    p = p * x2 + _TANH_ALPHA_7
    p = p * x2 + _TANH_ALPHA_5
    p = p * x2 + _TANH_ALPHA_3
    p = p * x2 + _TANH_ALPHA_1
    p *= x
    q = x2 * _TANH_BETA_6 + _TANH_BETA_4
    q = q * x2 + _TANH_BETA_2
    q = q * x2 + _TANH_BETA_0
    return p / q


def _lstm_direction(
    inputs: np.ndarray,
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
    bias: np.ndarray,
    *,
    reverse: bool,
) -> np.ndarray:
    seq_len = inputs.shape[0]
    output = np.empty((seq_len, HIDDEN_SIZE), dtype=np.float32)
    hidden = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    cell = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    input_gates = inputs @ input_weights
    indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

    for time_index in indices:
        gates = input_gates[time_index] + hidden @ recurrent_weights + bias
        input_gate, output_gate, forget_gate, cell_gate = np.split(gates, 4)
        input_gate = _logistic(input_gate)
        output_gate = _logistic(output_gate)
        forget_gate = _logistic(forget_gate)
        cell_gate = _tanh(cell_gate)
        cell = forget_gate * cell + input_gate * cell_gate
        hidden = output_gate * _tanh(cell)
        output[time_index] = hidden

    return output


def _bilstm_layer(
    inputs: np.ndarray,
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    forward = _lstm_direction(
        inputs,
        input_weights[0],
        recurrent_weights[0],
        bias[0],
        reverse=False,
    )
    backward = _lstm_direction(
        inputs,
        input_weights[1],
        recurrent_weights[1],
        bias[1],
        reverse=True,
    )
    return np.concatenate((forward, backward), axis=1)


def _crf_decode(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
) -> np.ndarray:
    if emissions.ndim != _EMISSION_NDIM or emissions.shape[1] != NUM_TAGS:
        message = f"unexpected emissions shape: {emissions.shape}"
        raise ValueError(message)
    seq_len = emissions.shape[0]
    if seq_len == 0:
        return np.empty(0, dtype=np.int32)

    score = start_transitions + emissions[0]
    history: list[np.ndarray] = []
    for time_index in range(1, seq_len):
        candidates = score[:, None] + transitions + emissions[time_index][None, :]
        history.append(np.argmax(candidates, axis=0))
        score = np.max(candidates, axis=0)

    tag = int(np.argmax(score + end_transitions))
    path = np.empty(seq_len, dtype=np.int32)
    path[-1] = tag
    for time_index in range(seq_len - 2, -1, -1):
        tag = int(history[time_index][tag])
        path[time_index] = tag
    return path


def _crf_decode_topk(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    seq_len, num_tags = emissions.shape
    if seq_len == 0:
        return []

    score = np.full((num_tags, k), -np.inf, dtype=np.float32)
    score[:, 0] = start_transitions + emissions[0]
    history: list[np.ndarray] = []

    for time_index in range(1, seq_len):
        candidates = score[:, :, None] + transitions[:, None, :] + emissions[time_index][None, None, :]
        candidates = candidates.reshape(num_tags * k, num_tags)
        indices = np.argsort(-candidates, axis=0)[:k]
        history.append(indices.T)
        score = np.take_along_axis(candidates, indices, axis=0).T

    final = score + end_transitions[:, None]
    flat = final.reshape(-1)
    paths: list[np.ndarray] = []
    for flat_index in np.argsort(-flat):
        if np.isneginf(flat[flat_index]):
            break
        tag, rank = divmod(int(flat_index), k)
        path = np.empty(seq_len, dtype=np.int32)
        path[-1] = tag
        for time_index in range(seq_len - 1, 0, -1):
            tag, rank = divmod(int(history[time_index - 1][tag, rank]), k)
            path[time_index - 1] = tag
        paths.append(path)
    return paths


def _decode_words(text: str, labels: np.ndarray) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char, label in zip(text, labels, strict=True):
        if int(label) == 1 and current:
            words.append("".join(current))
            current = [char]
        else:
            current.append(char)
    if current:
        words.append("".join(current))
    return words


class NumpyDKSplit:
    """NumPy-only inference for converted DKSplit weights."""

    def __init__(self, model_path: str | Path) -> None:
        with np.load(model_path) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                message = f"unsupported converted model format: {version}"
                raise ValueError(message)

            self.embedding = _dequantize(
                data["embedding_q"],
                data["embedding_scale"],
                data["embedding_zp"],
            )
            self.layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for layer in range(NUM_LAYERS):
                self.layers.append(
                    (
                        _dequantize(
                            data[f"l{layer}_w_q"],
                            data[f"l{layer}_w_scale"],
                            data[f"l{layer}_w_zp"],
                        ),
                        _dequantize(
                            data[f"l{layer}_r_q"],
                            data[f"l{layer}_r_scale"],
                            data[f"l{layer}_r_zp"],
                        ),
                        _combined_bias(np.asarray(data[f"l{layer}_bias"], dtype=np.float32)),
                    )
                )

            self.projection_weights = _dequantize(
                data["projection_q"],
                data["projection_scale"],
                data["projection_zp"],
            )
            self.projection_bias = np.asarray(data["projection_bias"], dtype=np.float32)
            self.transitions = np.asarray(data["crf_transitions"], dtype=np.float32)
            self.start_transitions = np.asarray(data["crf_start_transitions"], dtype=np.float32)
            self.end_transitions = np.asarray(data["crf_end_transitions"], dtype=np.float32)

        if self.embedding.shape != (VOCAB_SIZE, HIDDEN_SIZE):
            message = f"unexpected embedding shape: {self.embedding.shape}"
            raise ValueError(message)
        if self.projection_weights.shape != (NUM_DIRECTIONS * HIDDEN_SIZE, NUM_TAGS):
            message = f"unexpected projection shape: {self.projection_weights.shape}"
            raise ValueError(message)

    def emissions(self, text: str) -> np.ndarray:
        """Return two CRF emission scores for each processed character."""
        ids = text_to_ids(text)
        hidden = np.asarray(self.embedding[ids], dtype=np.float32)
        for input_weights, recurrent_weights, bias in self.layers:
            hidden = _bilstm_layer(hidden, input_weights, recurrent_weights, bias)
        return np.asarray(hidden @ self.projection_weights + self.projection_bias, dtype=np.float32)

    def split(self, text: str) -> list[str]:
        """Return the highest-scoring segmentation."""
        if not text:
            return []
        processed = text.lower()[:MAX_LEN]
        labels = _crf_decode(
            self.emissions(processed),
            self.transitions,
            self.start_transitions,
            self.end_transitions,
        )
        return _decode_words(processed, labels)

    def split_topk(self, text: str, k: int = 3) -> list[list[str]]:
        """Return up to ``k`` distinct segmentations, best first."""
        if k < 1:
            message = "k must be >= 1"
            raise ValueError(message)
        if not text:
            return []

        processed = text.lower()[:MAX_LEN]
        paths = _crf_decode_topk(
            self.emissions(processed),
            self.transitions,
            self.start_transitions,
            self.end_transitions,
            _UNIQUE_PATH_MULTIPLIER * k,
        )
        results: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for path in paths:
            words = _decode_words(processed, path)
            key = tuple(words)
            if key in seen:
                continue
            seen.add(key)
            results.append(words)
            if len(results) == k:
                break
        return results
