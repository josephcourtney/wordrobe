# DKSplit NumPy experiment

This directory contains only the exploratory runtime candidate for evaluating
DKSplit's fixed BiLSTM-CRF model without ONNX Runtime. It remains deliberately
outside `src/wordrobe` and is not part of Wordrobe's public API or dependency
surface.

## Runtime boundary

`runtime.py` is the only inference implementation. It depends on NumPy plus a
converted weights NPZ and provides:

- DKSplit-compatible lowercase/truncate/character preprocessing;
- the fixed three-layer bidirectional LSTM in float32;
- the 768 -> 2 emission projection;
- two-state CRF best-path decoding;
- k-best CRF decoding with deduplicated segmentations;
- `NumpyDKSplit.split()` and `NumpyDKSplit.split_topk()`.

It intentionally contains no ONNX parsing, ONNX Runtime integration, benchmark
downloads, model inspection, dynamic-quantization emulation, or test corpus
generation.

## Development scripts

All implementation/conversion/reference machinery lives under the repository's
`scripts/` directory:

- `scripts/convert_dksplit_weights.py` converts DKSplit's published ONNX model
  and CRF parameters into the compact NPZ consumed by `runtime.py`.
- `scripts/compare_dksplit_onnx.py` runs the minimal runtime against the real
  DKSplit ONNX Runtime implementation, including representative inputs and the
  published 1,000-domain benchmark.

Neither script is part of the inference dependency path.

## Recovered architecture

The published DKSplit 1.0.2 model uses:

- 38 character IDs (`PAD`, `UNK`, 26 letters, 10 digits);
- a `38 x 384` embedding;
- three bidirectional LSTM layers;
- hidden size 384 per direction;
- IOFC gate order;
- a 768 -> 2 emission projection;
- a two-state linear-chain CRF outside the ONNX graph.

The source model uses dynamically quantized ONNX Runtime operators. The runtime
candidate instead dequantizes the static weights once and evaluates the fixed
architecture in float32. A previous NumPy implementation that tried to emulate
dynamic activation quantization was substantially slower and did not improve
behavioral parity enough to justify keeping it, so it has been removed.

## Conversion

With DKSplit and ONNX installed for development:

```bash
python scripts/convert_dksplit_weights.py --output /tmp/dksplit-numpy.npz
```

Explicit source paths can also be supplied with `--onnx` and `--crf`.

The converted archive contains only the quantized embedding/LSTM/projection
weights, their scale/zero-point data, biases, format version, and CRF
parameters. The ONNX graph and operator metadata are discarded. For DKSplit
1.0.2, the source ONNX file is 9,538,484 bytes and the compressed converted
archive is about 7.05 MB. No model weights are committed to Wordrobe.

## Validation

The branch-only workflow performs the conversion and then runs:

```bash
python scripts/compare_dksplit_onnx.py --weights /tmp/dksplit-numpy.npz
```

On DKSplit's published 1,000-domain benchmark, the stripped float32 runtime has
shown:

- 99.7% top-1 parity with ONNX;
- strict exact match 86.4% vs ONNX 86.5%;
- lenient exact match 91.4% vs ONNX 91.5%;
- identical acceptable-result recall at top 3 (98.5%) and top 5 (99.3%).

The main tradeoff is speed: straightforward NumPy single-string inference is
roughly 8-9x slower than ONNX Runtime on the GitHub Linux runner.

## Licensing

The converted tensors are derived from the DKSplit model and remain subject to
its model licensing and attribution requirements. This experiment does not
vendor or redistribute those weights.
