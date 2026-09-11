# Experiment 5A: Identity-Initialized Parametric Refine

Experiment 5A is Experiment 4A with one controlled change: the final
convolution of every refine predictor is initialized to zero. Therefore, at
the first optimization step, `delta1 = 0`, `a2 = a3 = 1`, and `b2 = b3 = 0`;
the complete refine model exactly returns the frozen ConvGuided coarse image.

The five-layer parameterized cascade, delta supervision, BSDS400 full-image
training, geometry augmentation, optimizer, and 80-epoch schedule are
unchanged from Experiment 4A. Sigma 25 is the only supported noise level.

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp5
bash run_exp5a_sigma25.sh 1
```

Checkpoints use the `exp5a_identity_parametric_` prefix in `models/sigma25/`.
