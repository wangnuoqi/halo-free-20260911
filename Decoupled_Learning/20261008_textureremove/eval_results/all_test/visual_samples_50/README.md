# RTV visual samples

This directory contains 50 deterministic test samples selected by sorted
filename from the complete 1,154-image evaluation. Each sample has the same
stem followed by a variant suffix:

- `__input.png`: input image
- `__target.png`: fixed RTV operator target used for evaluation
- `__stage1.png`, `__stage2.png`, `__stage3.png`: our three-stage outputs
- `__official.png`: official DecoupleLearning single-operator baseline
- `__nafnet_patch.png`: NAFNet patch-256 baseline
- `__comparison.png`: existing labeled horizontal comparison, when available

The source evaluation used the shared full-image top-left crop to even height
and width, BGR input convention, FP32 inference, and the checkpoints recorded
in `../protocol.json`. The target is the generated RTV operator output, not a
human-authored ideal clean image.
