"""Optional NumPy inference for the fixed DKSplit boundary model."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised through optional dependency behavior
    msg = "neural boundary inference requires the 'neural' extra: pip install 'wordrobe[neural]'"
    raise ImportError(msg) from exc

if TYPE_CHECKING:
    from collections.abc import Sequence

CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"
UNK_IDX = 1
MAX_LEN = 64
HIDDEN_SIZE = 384
NUM_LAYERS = 3
NUM_DIRECTIONS = 2
NUM_TAGS = 2
FORMAT_VERSION = 1
VOCAB_SIZE = 38
MODEL_RESOURCE = "data/dksplit-boundaries-v1.npz"

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


class UnsupportedNeuralTextError(ValueError):
    """Raised when text falls outside the model's supported input domain."""


def supports_neural_text(text: str) -> bool:
    """Return whether *text* can be scored without truncation or unknown characters."""
    return bool(text) and len(text) <= MAX_LEN and text.isascii() and text.isalnum()


def _require_supported(text: str) -> str:
    if not supports_neural_text(text):
        msg = f"neural boundary model requires 1-{MAX_LEN} ASCII alphanumeric characters"
        raise UnsupportedNeuralTextError(msg)
    return text.lower()


def _text_to_ids(text: str) -> np.ndarray:
    processed = _require_supported(text)
    raw = np.frombuffer(processed.encode("ascii"), dtype=np.uint8)
    return _CHAR_MAP[raw]


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
        msg = f"unexpected LSTM bias shape: {bias.shape}; expected {expected}"
        raise ValueError(msg)
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
) -> tuple[int, ...]:
    seq_len = emissions.shape[0]
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
    return tuple(int(value) for value in path)


def _crf_decode_topk(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
    k: int,
) -> list[tuple[int, ...]]:
    seq_len, num_tags = emissions.shape
    score = np.full((num_tags, k), -np.inf, dtype=np.float32)
    score[:, 0] = start_transitions + emissions[0]
    history: list[np.ndarray] = []

    for time_index in range(1, seq_len):
        candidates = score[:, :, None] + transitions[:, None, :] + emissions[time_index][None, None, :]
        candidates = candidates.reshape(num_tags * k, num_tags)
        indices = np.argsort(-candidates, axis=0)[:k]
        history.append(indices.T)
        score = np.take_along_axis(candidates, indices, axis=0).T

    flat = (score + end_transitions[:, None]).reshape(-1)
    paths: list[tuple[int, ...]] = []
    for flat_index in np.argsort(-flat)[:k]:
        tag, rank = divmod(int(flat_index), k)
        path = np.empty(seq_len, dtype=np.int32)
        path[-1] = tag
        for time_index in range(seq_len - 1, 0, -1):
            tag, rank = divmod(int(history[time_index - 1][tag, rank]), k)
            path[time_index - 1] = tag
        paths.append(tuple(int(value) for value in path))
    return paths


class NeuralBoundaryModel:
    """NumPy-only inference for Wordrobe's converted DKSplit model resource."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            resource = resources.files("wordrobe").joinpath(MODEL_RESOURCE)
            with resource.open("rb") as stream:
                self._load(stream)
        else:
            with Path(model_path).open("rb") as stream:
                self._load(stream)

    def _load(self, stream: BinaryIO) -> None:
        with np.load(stream) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                msg = f"unsupported neural model format: {version}"
                raise ValueError(msg)

            self._embedding = _dequantize(
                data["embedding_q"],
                data["embedding_scale"],
                data["embedding_zp"],
            )
            self._layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for layer in range(NUM_LAYERS):
                self._layers.append((
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
                ))

            self._projection_weights = _dequantize(
                data["projection_q"],
                data["projection_scale"],
                data["projection_zp"],
            )
            self._projection_bias = np.asarray(data["projection_bias"], dtype=np.float32)
            self._transitions = np.asarray(data["crf_transitions"], dtype=np.float32)
            self._start_transitions = np.asarray(data["crf_start_transitions"], dtype=np.float32)
            self._end_transitions = np.asarray(data["crf_end_transitions"], dtype=np.float32)

        if self._embedding.shape != (VOCAB_SIZE, HIDDEN_SIZE):
            msg = f"unexpected embedding shape: {self._embedding.shape}"
            raise ValueError(msg)
        if self._projection_weights.shape != (NUM_DIRECTIONS * HIDDEN_SIZE, NUM_TAGS):
            msg = f"unexpected projection shape: {self._projection_weights.shape}"
            raise ValueError(msg)

    def emissions(self, text: str) -> np.ndarray:
        """Return CRF emission scores for each character in supported *text*."""
        ids = _text_to_ids(text)
        hidden = np.asarray(self._embedding[ids], dtype=np.float32)
        for input_weights, recurrent_weights, bias in self._layers:
            hidden = _bilstm_layer(hidden, input_weights, recurrent_weights, bias)
        return np.asarray(hidden @ self._projection_weights + self._projection_bias, dtype=np.float32)

    def best_labels(self, text: str) -> tuple[int, ...]:
        """Return the highest-scoring CRF label sequence."""
        emissions = self.emissions(text)
        return _crf_decode(
            emissions,
            self._transitions,
            self._start_transitions,
            self._end_transitions,
        )

    def topk_labels(self, text: str, k: int = 3) -> list[tuple[int, ...]]:
        """Return the ``k`` highest-scoring CRF label sequences."""
        if k < 1:
            msg = "k must be >= 1"
            raise ValueError(msg)
        emissions = self.emissions(text)
        return _crf_decode_topk(
            emissions,
            self._transitions,
            self._start_transitions,
            self._end_transitions,
            k,
        )

    def score_labels(self, text: str, labels: Sequence[int]) -> float:
        """Return the CRF score of one complete label sequence."""
        emissions = self.emissions(text)
        label_array = np.asarray(labels, dtype=np.int64)
        if label_array.shape != (len(text),):
            msg = f"expected {len(text)} labels, got shape {label_array.shape}"
            raise ValueError(msg)
        if np.any((label_array < 0) | (label_array >= NUM_TAGS)):
            msg = f"labels must be in [0, {NUM_TAGS - 1}]"
            raise ValueError(msg)

        score = self._start_transitions[label_array[0]] + emissions[0, label_array[0]]
        for index in range(1, len(text)):
            previous = label_array[index - 1]
            current = label_array[index]
            score += self._transitions[previous, current] + emissions[index, current]
        score += self._end_transitions[label_array[-1]]
        return float(score)


__all__ = [
    "MAX_LEN",
    "MODEL_RESOURCE",
    "NeuralBoundaryModel",
    "UnsupportedNeuralTextError",
    "supports_neural_text",
]
