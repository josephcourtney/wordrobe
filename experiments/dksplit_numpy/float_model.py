"""Practical float32 NumPy inference for the fixed DKSplit architecture.

This backend dequantizes DKSplit's static INT8 weights once at load time, then
uses NumPy/BLAS float32 matrix operations. It intentionally does not reproduce
ONNX Runtime's dynamic activation quantization; ``model.py`` retains that much
slower quantization-emulation path as a numerical reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model import (
    FORMAT_VERSION,
    HIDDEN_SIZE,
    MAX_LEN,
    NUM_DIRECTIONS,
    NUM_LAYERS,
    _combined_bias,
    _crf_decode,
    _decode_words,
    _dequantize,
    _mlas_logistic,
    _mlas_tanh,
    text_to_ids,
)


def _float_lstm_direction(
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
    time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

    for time_index in time_indices:
        gates = input_gates[time_index] + hidden @ recurrent_weights + bias
        input_gate, output_gate, forget_gate, cell_gate = np.split(gates, 4)
        input_gate = _mlas_logistic(input_gate)
        output_gate = _mlas_logistic(output_gate)
        forget_gate = _mlas_logistic(forget_gate)
        cell_gate = _mlas_tanh(cell_gate)
        cell = forget_gate * cell + input_gate * cell_gate
        hidden = output_gate * _mlas_tanh(cell)
        output[time_index] = hidden

    return output


def _float_bilstm_layer(
    inputs: np.ndarray,
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    forward = _float_lstm_direction(
        inputs,
        input_weights[0],
        recurrent_weights[0],
        bias[0],
        reverse=False,
    )
    backward = _float_lstm_direction(
        inputs,
        input_weights[1],
        recurrent_weights[1],
        bias[1],
        reverse=True,
    )
    return np.concatenate((forward, backward), axis=1)


class FloatNumpyDKSplit:
    """BLAS-backed float32 approximation of DKSplit inference."""

    def __init__(self, model_path: str | Path) -> None:
        with np.load(model_path) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"unsupported converted model format: {version}")

            self.embedding = _dequantize(data["embedding_q"], data["embedding_scale"], data["embedding_zp"])
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

        if self.embedding.shape != (38, HIDDEN_SIZE):
            raise ValueError(f"unexpected embedding shape: {self.embedding.shape}")
        if self.projection_weights.shape != (NUM_DIRECTIONS * HIDDEN_SIZE, 2):
            raise ValueError(f"unexpected projection shape: {self.projection_weights.shape}")

    def emissions_from_ids(self, char_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(char_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("char_ids must be one-dimensional")
        hidden = np.asarray(self.embedding[ids], dtype=np.float32)
        for input_weights, recurrent_weights, bias in self.layers:
            hidden = _float_bilstm_layer(hidden, input_weights, recurrent_weights, bias)
        return np.asarray(hidden @ self.projection_weights + self.projection_bias, dtype=np.float32)

    def emissions(self, text: str) -> np.ndarray:
        return self.emissions_from_ids(text_to_ids(text))

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        processed = text.lower()[:MAX_LEN]
        emissions = self.emissions(processed)
        labels = _crf_decode(emissions, self.transitions, self.start_transitions, self.end_transitions)
        return _decode_words(processed, labels)
