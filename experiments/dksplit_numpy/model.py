"""NumPy-only inference for the fixed DKSplit BiLSTM-CRF architecture.

This is an exploratory reimplementation, not part of wordrobe's public API.
The runtime consumes DKSplit's original INT8 weights and reproduces the
activation quantization used by ONNX Runtime's DynamicQuantizeLSTM and final
DynamicQuantizeLinear/MatMulInteger projection. The LSTM nonlinearities use
NumPy ports of ONNX Runtime MLAS's logistic and tanh rational approximations.
ONNX is needed only by the one-time converter and parity harness.
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

# MLAS logistic coefficients, copied from ONNX Runtime's logistic.cpp.
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

# MLAS tanh coefficients, copied from ONNX Runtime's tanh.cpp.
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
    """Apply DKSplit's lowercase/truncate/ASCII character mapping."""
    processed = text.lower()[:MAX_LEN]
    raw = np.frombuffer(processed.encode("ascii", errors="replace"), dtype=np.uint8)
    return _CHAR_MAP[np.clip(raw, 0, 127)]


def _dequantize(q: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
    """Dequantize a static tensor such as the character embedding table."""
    qf = np.asarray(q, dtype=np.float32)
    scale_f = np.asarray(scale, dtype=np.float32)
    zp_f = np.asarray(zero_point, dtype=np.float32)
    if scale_f.ndim == 0:
        return (qf - zp_f) * scale_f
    broadcast_shape = (scale_f.shape[0],) + (1,) * (qf.ndim - 1)
    return (qf - zp_f.reshape(broadcast_shape)) * scale_f.reshape(broadcast_shape)


def _dynamic_quantize_uint8(value: np.ndarray) -> tuple[np.ndarray, np.float32, np.uint8]:
    """Reproduce ONNX Runtime's uint8 GetQuantizationParameter + quantize."""
    x = np.asarray(value, dtype=np.float32)
    x_min = np.float32(min(0.0, float(np.min(x))))
    x_max = np.float32(max(0.0, float(np.max(x))))
    scale = np.float32(1.0 if x_max == x_min else (x_max - x_min) / 255.0)
    zero_point_float = np.float32(-x_min / scale)
    zero_point = np.uint8(np.clip(np.rint(zero_point_float), 0, 255))
    quantized = np.clip(np.rint(x / scale) + zero_point, 0, 255).astype(np.uint8)
    return quantized, scale, zero_point


def _quantized_matmul_from_activation(
    activation: np.ndarray,
    weight_q: np.ndarray,
    weight_scale: np.float32,
    weight_zero_point: np.generic | int,
) -> np.ndarray:
    """Run one dynamically-quantized activation x statically-quantized weight GEMM."""
    activation_q, activation_scale, activation_zero_point = _dynamic_quantize_uint8(activation)
    left = activation_q.astype(np.int32) - np.int32(activation_zero_point)
    right = np.asarray(weight_q, dtype=np.int32) - np.int32(weight_zero_point)
    accumulator = left @ right
    output_scale = np.float32(activation_scale * np.float32(weight_scale))
    return np.asarray(accumulator, dtype=np.float32) * output_scale


def _quantized_projection(
    hidden: np.ndarray,
    weight_q: np.ndarray,
    weight_scale: np.float32,
    weight_zero_point: np.generic | int,
    bias: np.ndarray,
) -> np.ndarray:
    return _quantized_matmul_from_activation(hidden, weight_q, weight_scale, weight_zero_point) + bias


def _mlas_logistic(value: np.ndarray) -> np.ndarray:
    """Port MLAS's clamped rational logistic approximation."""
    x = np.clip(np.asarray(value, dtype=np.float32), np.float32(-18.0), np.float32(18.0))
    x2 = x * x
    p = x2 * _LOGISTIC_ALPHA_9 + _LOGISTIC_ALPHA_7
    p = p * x2 + _LOGISTIC_ALPHA_5
    p = p * x2 + _LOGISTIC_ALPHA_3
    p = p * x2 + _LOGISTIC_ALPHA_1
    p = p * x
    q = x2 * _LOGISTIC_BETA_10 + _LOGISTIC_BETA_8
    q = q * x2 + _LOGISTIC_BETA_6
    q = q * x2 + _LOGISTIC_BETA_4
    q = q * x2 + _LOGISTIC_BETA_2
    q = q * x2 + _LOGISTIC_BETA_0
    return np.clip(p / q + np.float32(0.5), np.float32(0.0), np.float32(1.0))


def _mlas_tanh(value: np.ndarray) -> np.ndarray:
    """Port MLAS's clamped rational tanh approximation."""
    x = np.clip(np.asarray(value, dtype=np.float32), np.float32(-9.0), np.float32(9.0))
    x2 = x * x
    p = x2 * _TANH_ALPHA_13 + _TANH_ALPHA_11
    p = p * x2 + _TANH_ALPHA_9
    p = p * x2 + _TANH_ALPHA_7
    p = p * x2 + _TANH_ALPHA_5
    p = p * x2 + _TANH_ALPHA_3
    p = p * x2 + _TANH_ALPHA_1
    p = p * x
    q = x2 * _TANH_BETA_6 + _TANH_BETA_4
    q = q * x2 + _TANH_BETA_2
    q = q * x2 + _TANH_BETA_0
    return p / q


def _combined_bias(bias: np.ndarray) -> np.ndarray:
    """Combine ONNX Wb[iofc] and Rb[iofc] halves."""
    if bias.shape != (NUM_DIRECTIONS, 8 * HIDDEN_SIZE):
        raise ValueError(f"unexpected LSTM bias shape: {bias.shape}")
    return bias[:, : 4 * HIDDEN_SIZE] + bias[:, 4 * HIDDEN_SIZE :]


def _lstm_direction(
    inputs: np.ndarray,
    input_weight_q: np.ndarray,
    input_weight_scale: np.float32,
    input_weight_zero_point: np.generic | int,
    recurrent_weight_q: np.ndarray,
    recurrent_weight_scale: np.float32,
    recurrent_weight_zero_point: np.generic | int,
    bias: np.ndarray,
    *,
    reverse: bool,
) -> np.ndarray:
    """Run one DynamicQuantizeLSTM direction for a single sequence.

    ONNX Runtime quantizes the full input matrix once for X@W, then quantizes
    the previous hidden state independently for every H@R recurrence. DKSplit's
    optimized weights are laid out [input, 4H] and [H, 4H], in IOFC gate order.
    """
    seq_len = inputs.shape[0]
    output = np.empty((seq_len, HIDDEN_SIZE), dtype=np.float32)
    hidden = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    cell = np.zeros(HIDDEN_SIZE, dtype=np.float32)

    input_gates = _quantized_matmul_from_activation(
        inputs,
        input_weight_q,
        input_weight_scale,
        input_weight_zero_point,
    )
    time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

    for time_index in time_indices:
        recurrent_gates = _quantized_matmul_from_activation(
            hidden.reshape(1, -1),
            recurrent_weight_q,
            recurrent_weight_scale,
            recurrent_weight_zero_point,
        )[0]
        gates = input_gates[time_index] + recurrent_gates + bias
        input_gate, output_gate, forget_gate, cell_gate = np.split(gates, 4)
        input_gate = _mlas_logistic(input_gate)
        output_gate = _mlas_logistic(output_gate)
        forget_gate = _mlas_logistic(forget_gate)
        cell_gate = _mlas_tanh(cell_gate)
        cell = forget_gate * cell + input_gate * cell_gate
        hidden = output_gate * _mlas_tanh(cell)
        output[time_index] = hidden

    return output


def _bilstm_layer(
    inputs: np.ndarray,
    input_weight_q: np.ndarray,
    input_weight_scale: np.ndarray,
    input_weight_zero_point: np.ndarray,
    recurrent_weight_q: np.ndarray,
    recurrent_weight_scale: np.ndarray,
    recurrent_weight_zero_point: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    forward = _lstm_direction(
        inputs,
        input_weight_q[0],
        np.float32(input_weight_scale[0]),
        input_weight_zero_point[0],
        recurrent_weight_q[0],
        np.float32(recurrent_weight_scale[0]),
        recurrent_weight_zero_point[0],
        bias[0],
        reverse=False,
    )
    backward = _lstm_direction(
        inputs,
        input_weight_q[1],
        np.float32(input_weight_scale[1]),
        input_weight_zero_point[1],
        recurrent_weight_q[1],
        np.float32(recurrent_weight_scale[1]),
        recurrent_weight_zero_point[1],
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
            self.layers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for layer in range(NUM_LAYERS):
                self.layers.append(
                    (
                        np.asarray(data[f"l{layer}_w_q"]),
                        np.asarray(data[f"l{layer}_w_scale"], dtype=np.float32),
                        np.asarray(data[f"l{layer}_w_zp"]),
                        np.asarray(data[f"l{layer}_r_q"]),
                        np.asarray(data[f"l{layer}_r_scale"], dtype=np.float32),
                        np.asarray(data[f"l{layer}_r_zp"]),
                        _combined_bias(np.asarray(data[f"l{layer}_bias"], dtype=np.float32)),
                    )
                )

            self.projection_q = np.asarray(data["projection_q"])
            self.projection_scale = np.float32(data["projection_scale"])
            self.projection_zp = np.asarray(data["projection_zp"]).item()
            self.projection_bias = np.asarray(data["projection_bias"], dtype=np.float32)
            self.transitions = np.asarray(data["crf_transitions"], dtype=np.float32)
            self.start_transitions = np.asarray(data["crf_start_transitions"], dtype=np.float32)
            self.end_transitions = np.asarray(data["crf_end_transitions"], dtype=np.float32)

        if self.embedding.shape != (38, 384):
            raise ValueError(f"unexpected embedding shape: {self.embedding.shape}")
        if self.projection_q.shape != (768, 2):
            raise ValueError(f"unexpected projection shape: {self.projection_q.shape}")

    def emissions_from_ids(self, char_ids: np.ndarray) -> np.ndarray:
        """Return emission scores for one character-ID sequence."""
        ids = np.asarray(char_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("char_ids must be one-dimensional")
        hidden = np.asarray(self.embedding[ids], dtype=np.float32)
        for layer in self.layers:
            hidden = _bilstm_layer(hidden, *layer)
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
