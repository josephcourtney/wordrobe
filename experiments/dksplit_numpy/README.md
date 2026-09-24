# DKSplit NumPy experiment

This directory is an exploratory, non-integrated reimplementation of the fixed
DKSplit inference graph using NumPy. It is deliberately outside `src/wordrobe`
and is not part of Wordrobe's public API or runtime dependencies.

The experiment answers a narrow question: can DKSplit's trained model be used
without ONNX Runtime if its fixed architecture is implemented directly?

## Recovered architecture

Inspection of the shipped `dksplit-int8.onnx` graph shows:

- character vocabulary: 38 IDs (`PAD`, `UNK`, 26 letters, 10 digits)
- embedding: `38 x 384`, statically quantized `uint8`
- three bidirectional recurrent layers
- each direction has hidden size 384
- layer 0 input width: 384
- layers 1 and 2 input width: 768
- recurrent operator: ONNX Runtime `com.microsoft::DynamicQuantizeLSTM`
- gate order: IOFC
- final projection: 768 -> 2 tags, dynamically quantized activation with an
  `int8` static weight matrix
- two-state linear-chain CRF decoded outside the ONNX graph

## Files

- `inspect_onnx.py` summarizes the actual shipped graph and tensor layout.
- `convert.py` is the one-time converter from DKSplit's ONNX model plus CRF NPZ
  into a compact runtime NPZ.
- `float_model.py` is the practical NumPy implementation. It dequantizes static
  weights once and performs float32 BLAS-backed inference.
- `model.py` contains a much slower attempt to emulate ONNX Runtime's dynamic
  activation quantization and MLAS nonlinearities. It is retained as a
  diagnostic reference, not the preferred backend.
- `compare.py` compares emissions, top-1 segmentation, size, and runtime against
  the installed DKSplit/ONNX Runtime implementation.
- `realistic_compare.py` generates identifier-like inputs and compares top-1 and
  top-3 behavior.
- `benchmark_compare.py` evaluates both backends on DKSplit's published
  `sample_1000.csv` benchmark.

## Conversion

The converted archive intentionally contains only tensors needed by the fixed
architecture: quantized embedding/LSTM/projection weights, their scales and
zero points, biases, and CRF parameters. The graph, operator metadata, and ONNX
container are discarded.

Example:

```bash
python experiments/dksplit_numpy/convert.py \
  --onnx /path/to/dksplit-int8.onnx \
  --crf /path/to/dksplit.npz \
  --output /tmp/dksplit-numpy.npz
```

On the current DKSplit 1.0.2 model, the ONNX file is 9,538,484 bytes and the
compressed converted archive is about 7.05 MB. No converted model weights are
committed to Wordrobe.

## Numerical strategy

The practical backend does **not** attempt bit-for-bit reproduction of ONNX
Runtime's dynamic activation quantization. Instead it dequantizes DKSplit's
static weights once to float32 and evaluates the same fixed BiLSTM and CRF.
This makes the implementation simple and inspectable while retaining the
trained model.

A second backend in `model.py` emulates dynamic quantization more literally.
It is substantially slower in NumPy and still does not reproduce every
kernel-level numerical detail of ONNX Runtime, so it is useful primarily for
understanding the residual differences.

## Validation

The branch-only GitHub Actions workflow installs the real DKSplit package and
uses ONNX Runtime as an oracle. It checks:

1. character preprocessing parity;
2. emission differences;
3. top-1 CRF segmentation parity on representative and randomized strings;
4. top-1/top-3 parity on identifier-like synthetic data;
5. behavior on DKSplit's published 1,000-prefix benchmark;
6. rough single-string runtime and serialized model size.

The workflow is exploratory and the draft PR is intentionally not intended for
merge.

## Licensing

The converted tensors are derived from the DKSplit model and remain subject to
DKSplit/model licensing and attribution requirements. The experiment does not
vendor or redistribute those weights in this repository.
