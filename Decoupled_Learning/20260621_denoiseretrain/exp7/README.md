# Exp7: Unified Full-Image Protocol

Exp7 is the six-arm full-image comparison. Every RF run uses BSDS400 full
images, 8-way geometry augmentation, online sigma25 AWGN without noisy-input
clamping, full-image batch 8 by default, and 160 epochs. It evaluates complete BSD68 images
with the fixed NumPy `RandomState(0)` protocol.

`run_exp7.py` stages DnCNN/FFDNet initializers under `exp7/models/init/`, then
fine-tunes three independent full-image bases under `exp7/models/`:
ConvGuided, DnCNN, and FFDNet. Each corresponding frozen/joint RF run loads
only that local base checkpoint. Joint runs use Exp7C RF plus
`0.5 * SmoothL1(coarse, clean)`; frozen runs use Exp6D and update RF only.
