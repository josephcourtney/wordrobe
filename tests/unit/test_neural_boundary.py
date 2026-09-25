from __future__ import annotations

from importlib import resources

import pytest

np = pytest.importorskip("numpy")
neural = pytest.importorskip("wordrobe._neural_boundary")
MAX_LEN = neural.MAX_LEN
MODEL_RESOURCE = neural.MODEL_RESOURCE
NeuralBoundaryModel = neural.NeuralBoundaryModel
UnsupportedNeuralTextError = neural.UnsupportedNeuralTextError
supports_neural_text = neural.supports_neural_text


@pytest.fixture(scope="module")
def model() -> NeuralBoundaryModel:
    return NeuralBoundaryModel()


def _model_arrays() -> dict[str, object]:
    resource = resources.files("wordrobe").joinpath(MODEL_RESOURCE)
    with resource.open("rb") as stream, np.load(stream) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


@pytest.mark.parametrize(
    ("text", "supported"),
    [
        ("chatgptlogin", True),
        ("ipv6address", True),
        ("a" * MAX_LEN, True),
        ("", False),
        ("foo-bar", False),
        ("café", False),
        ("a" * (MAX_LEN + 1), False),
    ],
)
def test_supports_neural_text(text: str, *, supported: bool) -> None:
    assert supports_neural_text(text) is supported


def test_emissions_have_two_scores_per_character(model: NeuralBoundaryModel) -> None:
    emissions = model.emissions("chatgptlogin")

    assert emissions.shape == (len("chatgptlogin"), 2)
    assert emissions.dtype == np.float32
    assert np.isfinite(emissions).all()


def test_best_labels_are_first_topk_path(model: NeuralBoundaryModel) -> None:
    text = "expertsexchange"

    best = model.best_labels(text)
    candidates = model.topk_labels(text, 5)

    assert candidates[0] == best
    assert len(best) == len(text)
    assert all(label in {0, 1} for label in best)
    assert len(set(candidates)) == len(candidates)


def test_topk_paths_are_sorted_by_crf_score(model: NeuralBoundaryModel) -> None:
    text = "thisisnotable"
    candidates = model.topk_labels(text, 5)
    scores = [model.score_labels(text, labels) for labels in candidates]

    assert scores == sorted(scores, reverse=True)


def test_model_is_case_insensitive(model: NeuralBoundaryModel) -> None:
    assert model.best_labels("macOSVersion") == model.best_labels("macosversion")


@pytest.mark.parametrize("text", ["", "foo-bar", "café", "a" * (MAX_LEN + 1)])
def test_inference_rejects_unsupported_inputs(model: NeuralBoundaryModel, text: str) -> None:
    with pytest.raises(UnsupportedNeuralTextError):
        model.emissions(text)


def test_score_labels_validates_shape_and_values(model: NeuralBoundaryModel) -> None:
    with pytest.raises(ValueError, match="expected 4 labels"):
        model.score_labels("test", [0, 1])
    with pytest.raises(ValueError, match="labels must be"):
        model.score_labels("test", [0, 0, 0, 2])


def test_topk_requires_positive_k(model: NeuralBoundaryModel) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        model.topk_labels("test", 0)


def test_model_rejects_unknown_resource_format(tmp_path) -> None:
    arrays = _model_arrays()
    arrays["format_version"] = np.asarray(999, dtype=np.int32)
    path = tmp_path / "wrong-version.npz"
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="unsupported neural model format"):
        NeuralBoundaryModel(path)


def test_model_rejects_wrong_embedding_shape(tmp_path) -> None:
    arrays = _model_arrays()
    arrays["embedding_q"] = arrays["embedding_q"][:-1]
    path = tmp_path / "wrong-shape.npz"
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="unexpected embedding shape"):
        NeuralBoundaryModel(path)


def test_model_rejects_missing_required_tensor(tmp_path) -> None:
    arrays = _model_arrays()
    arrays.pop("projection_bias")
    path = tmp_path / "missing-tensor.npz"
    np.savez_compressed(path, **arrays)

    with pytest.raises(KeyError):
        NeuralBoundaryModel(path)


def test_model_rejects_corrupt_npz(tmp_path) -> None:
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a numpy archive")

    with pytest.raises(ValueError, match="pickled .* data"):
        NeuralBoundaryModel(path)


def test_model_reports_missing_explicit_resource(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        NeuralBoundaryModel(tmp_path / "missing.npz")
