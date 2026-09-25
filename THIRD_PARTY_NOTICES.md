# Third-party notices

## DKSplit model

Wordrobe's optional neural word-boundary model is derived from **DKSplit** by
ABTdomain:

- Project: https://github.com/ABTdomain/dksplit
- Source model version used for conversion: DKSplit 1.0.2
- Copyright: 2026 ABTdomain
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License text: https://creativecommons.org/licenses/by/4.0/legalcode

Wordrobe redistributes a transformed copy of the trained DKSplit model weights.
The original ONNX graph and CRF parameter archive are converted into a compact
NPZ containing only the tensors required by Wordrobe's fixed NumPy inference
implementation. The model architecture is reimplemented directly in NumPy;
ONNX Runtime is not required for Wordrobe inference.

These changes are adaptations of the DKSplit model and are not endorsed by
ABTdomain.
