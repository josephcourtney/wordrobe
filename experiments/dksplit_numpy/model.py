"""NumPy-only inference for the fixed DKSplit BiLSTM-CRF architecture.

This is an exploratory reimplementation, not part of wordrobe's public API.
The converted archive contains DKSplit's original quantized weights. LSTM
weights are dequantized to float32 once at load time; the final projection
reproduces ONNX's DynamicQuantizeLinear + MatMulInteger path directly. The
remaining numerical difference from ONNX Runtime is therefore confined to the
three DynamicQuantizeLSTM operators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"
PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 64
HIDDEN_SIZE = 384
NUM_LAYERS = 3
NUM_DIRECTIONS = 2
NUM_TAGS = 2
FORMAT_VERSION = 1

_CHAR_MAP = np.full(128, UNK_IDX, dtype=np.int64)
for _index, _char in enumerate(CHAR_VOCAB, start=2):
    _CHAR_MAP[ord(_char)] = _index


def text_to_ids(text: str) -> np.ndarray:
    """Apply DKSplit's lowercase/truncate/ASCII character mapping."""
    processed = text.lower()[:MAX_LEN]
    raw = np.frombuffer(processed.encode("ascii", errors="replace"), dtype=np.uint8)
    return _CHAR_MAP[np.clip(raw, 0, 127)]


def _dequantize(q: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
    qf = np.asarray(q, dtype=np.float32)
    scale_f = np.asarray(scale, dtype=np.float32)
    zp_f = np.asarray(zero_point, dtype=np.float32)
    if scale_f.ndim == 0:
        return (qf - zp_f) * scale_f
    broadcast_shape = (scale_f.shape[0],) + (1,) * (qf.ndim - 1)
    return (qf - zp_f.reshape(broadcast_shape)) * scale_f.reshape(broadcast_shape)


def _dynamic_quantize_uint8(value: np.ndarray) -> tuple[np.ndarray, np.float32, np.uint8]:
    """Reproduce ONNX DynamicQuantizeLinear's per-tensor uint8 transform."""
    x = np.asarray(value, dtype=np.float32)
    x_min = np.float32(min(0.0, float(np.min(x))))
    x_max = np.float32(max(0.0, float(np.max(x))))
    scale = np.float32((x_max - x_min) / 255.0)
    if scale == 0.0:
        scale = np.float32(1.0)
    zero_point_float = np.float32(-x_min / scale)
    zero_point = np.uint8(np.clip(np.rint(zero_point_float), 0, 255))
    quantized = np.clip(np.rint(x / scale) + zero_point, 0, 255).astype(np.uint8)
    return quantized, scale, zero_point


def _quantized_projection(
    hidden: np.ndarray,
    weight_q: np.ndarray,
    weight_scale: np.float32,
    weight_zero_point: np.int8,
    bias: np.ndarray,
) -> np.ndarray:
    """Reproduce the ONNX DynamicQuantizeLinear/MatMulInteger projection."""
    hidden_q, hidden_scale, hidden_zero_point = _dynamic_quantize_uint8(hidden)
    left = hidden_q.astype(np.int32) - np.int32(hidden_zero_point)
    right = np.asarray(weight_q, dtype=np.int32) - np.int32(weight_zero_point)
    integer_product = left @ right
    output_scale = np.float32(hidden_scale * weight_scale)
    return np.asarray(integer_product, dtype=np.float32) * output_scale + bias


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _combined_bias(bias: np.ndarray) -> np.ndarray:
    """Combine ONNX Wb[iofc] and Rb[iofc] halves."""
    if bias.shape != (NUM_DIRECTIONS, 8 * HIDDEN_SIZE):
        raise ValueError(f"unexpected LSTM bias shape: {bias.shape}")
    return bias[:, : 4 * HIDDEN_SIZE] + bias[:, 4 * HIDDEN_SIZE :]


def _lstm_direction(
    inputs: np.ndarray,
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
    bias: np.ndarray,
    *,
    reverse: bool,
) -> np.ndarray:
    """Run one ONNX-layout LSTM direction for a single sequence.

    DKSplit's dynamically-quantized ONNX graph stores the optimized matrices
    transposed relative to the standard ONNX tensor description: [input, 4H]
    and [H, 4H]. Gate chunks are ONNX IOFC order.
    """
    seq_len = inputs.shape[0]
    output = np.empty((seq_len, HIDDEN_SIZE), dtype=np.float32)
    hidden = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    cell = np.zeros(HIDDEN_SIZE, dtype=np.float32)

    # The input contribution is independent of recurrent state; doing it in
    # one GEMM substantially reduces Python/BLAS call overhead.
    input_gates = inputs @ input_weights + bias
    time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

    for time_index in time_indices:
        gates = input_gates[time_index] + hidden @ recurrent_weights
        input_gate, output_gate, forget_gate, cell_gate = np.split(gates, 4)
        input_gate = _sigmoid(input_gate)
        output_gate = _sigmoid(output_gate)
        forget_gate = _sigmoid(forget_gate)
        cell_gate = np.tanh(cell_gate)
        cell = forget_gate * cell + input_gate * cell_gate
        hidden = output_gate * np.tanh(cell)
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
    """Decode the best two-state linear-chain CRF path."""
    if emissions.ndim != 2 or emissions.shape[1] != NUM_TAGS:
        raise ValueError(f"unexpected emissions shape: {emissions.shape}")
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
    """Fixed-architecture NumPy inference engine for converted DKSplit weights."""

    def __init__(self, model_path: str | Path) -> None:
        with np.load(model_path) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"unsupported converted model format: {version}")

            self.embedding = _dequantize(data["embedding_q"], data["embedding_scale"], data["embedding_zp"])
            self.layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for layer in range(NUM_LAYERS):
                w = _dequantize(data[f"l{layer}_w_q"], data[f"l{layer}_w_scale"], data[f"l{layer}_w_zp"])
                r = _dequantize(data[f"l{layer}_r_q"], data[f"l{layer}_r_scale"], data[f"l{layer}_r_zp"])
                b = _combined_bias(np.asarray(data[f"l{layer}_bias"], dtype=np.float32))
                self.layers.append((w, r, b))

            self.projection_q = np.asarray(data["projection_q"], dtype=np.int8)
            self.projection_scale = np.float32(data["projection_scale"])
            self.projection_zp = np.int8(data["projection_zp"])
            self.projection_bias = np.asarray(data["projection_bias"], dtype=np.float32)
            self.transitions = np.asarray(data["crf_transitions"], dtype=np.float32)
            self.start_transitions = np.asarray(data["crf_start_transitions"], dtype=np.float32)
            self.end_transitions = np.asarray(data["crf_end_transitions"], dtype=np.float32)

        if self.embedding.shape != (38, 384):
            raise ValueError(f"unexpected embedding shape: {self.embedding.shape}")
        if self.projection_q.shape != (768, 2):
            raise ValueError(f"unexpected projection shape: {self.projection_q.shape}")

    def emissions_from_ids(self, char_ids: np.ndarray) -> np.ndarray:
        """Return float32 emission scores for one character-ID sequence."""
        ids = np.asarray(char_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("char_ids must be one-dimensional")
        hidden = np.asarray(self.embedding[ids], dtype=np.float32)
        for input_weights, recurrent_weights, bias in self.layers:
            hidden = _bilstm_layer(hidden, input_weights, recurrent_weights, bias)
        return np.asarray(
            _quantized_projection(
                hidden,
                self.projection_q,
                self.projection_scale,
                self.projection_zp,
                self.projection_bias,
            ),
            dtype=np.float32,
        )

    def emissions(self, text: str) -> np.ndarray:
        """Return emissions after DKSplit-compatible text preprocessing."""
        return self.emissions_from_ids(text_to_ids(text))

    def split(self, text: str) -> list[str]:
        """Return the best CRF segmentation, matching DKSplit's text normalization."""
        if not text:
            return []
        processed = text.lower()[:MAX_LEN]
        emissions = self.emissions(processed)
        labels = _crf_decode(emissions, self.transitions, self.start_transitions, self.end_transitions)
        return _decode_words(processed, labels)
