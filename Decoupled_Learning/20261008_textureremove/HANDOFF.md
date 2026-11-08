# Texture Removal Research and Experiment Handoff

## Scope

This handoff records the current operator-imitation experiment in this directory, followed by the literature review that motivated it. The experiment is a three-stage RGB model trained to imitate fixed-parameter RTV smoothing on VOC2012 trainval images. It does not establish that the output is a perceptually correct or human-verified texture-free image.

## Current Status

### Verified from code and artifacts

- Both RTV datasets exist under `/mnt/d/ResearchCode/halo-free/datasets/VOC2012/`: `RTV_fixed_lambda_0p01` (Y channel) and `RTV_RGB_fixed_lambda_0p01` (RGB). Each has 9,232 train, 1,154 validation, and 1,154 test input/label pairs, for 11,540 pairs total. Their `metadata.jsonl` files record source image, split, mode, operator, parameters, and paired paths.
- The active training code is configured for RGB: all three training scripts default to `RTV_RGB_fixed_lambda_0p01`; model constructors default to 3 channels; and the saved base and joint checkpoints have 3-channel input weights. RGB dataset metadata records `mode: rgb`. This establishes the model configuration as RGB. The exact dataset path used at launch is not independently recorded because `--data-root` can override the default and no invocation log was found. The Y dataset exists but is not the default input. The README's statement that the current scripts use the Y dataset is stale/inconsistent with the code.
- Stage 1 base training reached epoch 80. `models/conv_base_rtv/conv_base_rtv_full.pth` records `epoch=80` and `best_loss=0.00046228582666229206`; `conv_base_rtv_best.pth` is present.
- Stage 2 refine training reached epoch 80. `models/refine_rtv/refine_rtv_full.pth` records `epoch=80` and `best_loss=0.00012369336343237317`; `refine_rtv_best.pth` is present.
- A Stage 3 joint best checkpoint exists at `models/joint_rtv/joint_rtv_best.pth`. The training script saves only `base` and `refine` weights when validation improves. No joint epoch, validation score, optimizer state, or history log is available, so its training duration and best validation metric are unresolved.
- The historical official DecoupleLearning single-operator RTV baseline is crop-trained: `baseline/runs/rtv/rtv_official_best.pth` and `rtv_official_latest.pth` exist, both at epoch 100; best validation MSE is 8.70512058 in `[0,255]` scale. Its state loads strictly and has finite parameters.
- On all 1,154 RGB test pairs with the shared centered 256x256 crop, mean per-image MSE / MAE in `[0,1]` were: Stage 2 `0.000165254 / 0.007849425`; joint `0.000153019 / 0.007505718`; historical official baseline `0.000144290 / 0.007174591`. This compares inference on the same crop but not matching training/resolution protocols.
- On the first 16 full-resolution RGB test images, MSE/MAE were `0.000131278 / 0.006518074` for Stage 2, `0.000120763 / 0.006220614` for joint, and `0.010958159 / 0.080615781` for the crop-trained official baseline. This indicates its direct full-image inference is poorly matched to its crop training, but this is only a 16-image check.
- The aligned full-image official baseline has since completed training: `baseline/runs/rtv_full_image/rtv_official_latest.pth` records epoch 100, and `rtv_official_best.pth` records best epoch 99 with validation MSE `5.8000619387` on the official `[0,255]` scale. Its checkpoint protocol records batch size 8, batch-level top-left even-size trimming for training, and full-image validation.
- A fresh CPU evaluation compared `baseline/runs/rtv_full_image/rtv_official_best.pth` against `models/joint_rtv/joint_rtv_best.pth` on all 1,154 RGB test pairs at native full-image resolution (per-image MSE/MAE averaged across images, values normalized to `[0,1]`). Aligned official baseline: MSE `0.0000851953`, MAE `0.00535149`. Three-stage joint: MSE `0.0001083351`, MAE `0.00631366`. The aligned official baseline is lower by 21.4% MSE and 15.2% MAE in this operator-target fidelity evaluation. Inputs use the same BGR tensor convention, and odd bottom/right borders are trimmed consistently. This is a direct test-set comparison; the joint checkpoint does not preserve its training epoch or validation metric.
- `baseline/train_rtv.py` and its data loader have been changed to match the three-stage data protocol: train batch size 8, shuffle, `drop_last=False`, batch-level top-left trimming to the smallest even height/width, the same BGR channel order produced by the three-stage OpenCV loader, and full-image validation at batch size 1. Its official model-specific scaling, centering, and edge channel remain distinct. New checkpoints use `baseline/runs/rtv_full_image/` and store train/validation geometry plus batch size; resume rejects a mismatched protocol. The aligned run completed as recorded above. The old crop checkpoint remains a separate historical result.
- **Verified (2026-10-26):** The NAFNet RTV patch run reached its configured 200,000/200,000 optimizer updates. `baseline/runs/nafnet_rtv_patch256_3gpu/nafnet_rtv_latest.pth` and `nafnet_rtv_best.pth` both record `iter=200000`, `epoch=519`, and best full-image validation mean per-image MSE `0.000030062025568268922` on `[0,1]`. The recorded protocol is random paired 256x256 training crops, full-image validation, BGR `[0,1]`, three GPUs, batch 8 per GPU (effective 24), BF16 forward, AdamW, and a 200,000-update cosine schedule. Both files were last modified on 2026-10-25 at 11:33 local time. No NAFNet training process was running at the 2026-10-26 check. This is the completed batch-8 run; no batch-16 run artifacts were found. A NAFNet test-split evaluation against the three-stage joint model has not yet been recorded, so this validation MSE is not a test comparison.

### Implemented but not verified as run

- `baseline/train_rtv.py` is the implementation for the official-model comparison and its checkpoints are verified above; no separate text training log was found.
- The Python dataset generator is a prototype; the documented full VOC2012 RTV generation path is the MATLAB generator and its `run_generate_rtv_voc2012*.m` entry points.

## Verified Artifacts

- Y dataset: `/mnt/d/ResearchCode/halo-free/datasets/VOC2012/RTV_fixed_lambda_0p01/metadata.jsonl` reports `mode: y` and the fixed RTV settings.
- RGB dataset: `/mnt/d/ResearchCode/halo-free/datasets/VOC2012/RTV_RGB_fixed_lambda_0p01/metadata.jsonl` reports `mode: rgb` and the same settings. Both dataset roots contain train/val/test input and label directories with matching counts.
- Checkpoints: `models/conv_base_rtv/{conv_base_rtv_best.pth,conv_base_rtv_full.pth}`, `models/refine_rtv/{refine_rtv_best.pth,refine_rtv_full.pth}`, and `models/joint_rtv/joint_rtv_best.pth`.
- Historical crop baseline: `baseline/runs/rtv/rtv_official_{best,latest}.pth`; epoch 100, `rtv_lambda=0.01`, condition `0.035`, validation MSE `8.7051205824` on `[0,255]` scale.
- Aligned full-image baseline: `baseline/runs/rtv_full_image/rtv_official_latest.pth` (epoch 100) and `rtv_official_best.pth` (best epoch 99) are present. The best checkpoint's full-image validation MSE is `5.8000619387` on `[0,255]`. Full test set per-image mean MSE/MAE on `[0,1]` are `0.0000851953 / 0.00535149`.
- NAFNet patch-256 baseline: `baseline/runs/nafnet_rtv_patch256_3gpu/nafnet_rtv_{latest,best}.pth`; both record 200,000 updates and best validation MSE `0.000030062025568268922` on `[0,1]`. This is validation evidence only; no NAFNet full-test metric is recorded.
- Channel convention: MATLAB TIFF data is RGB, but `cv2.imread` in the three-stage dataset returns BGR. The aligned baseline reader explicitly reverses TIFF channels to match. Predictions are in that same BGR order; any RGB image export/viewer must reverse channels back to RGB. Metrics against labels are unaffected by the shared channel permutation.
- Historical shared-center-crop comparison: all 1,154 RGB test pairs, `256x256` crops, metrics in `[0,1]`: Stage 2 MSE/MAE `0.000165254 / 0.007849425`; joint `0.000153019 / 0.007505718`; crop-trained official baseline `0.000144290 / 0.007174591`.
- Full-image test comparison: all 1,154 RGB test pairs, native full-image inference with consistent even-size top-left trimming; per-image mean MSE/MAE in `[0,1]`: three-stage joint `0.0001083351 / 0.00631366`; aligned official baseline best checkpoint `0.0000851953 / 0.00535149`. The evaluation was run on CPU because CUDA tensor allocation in the WSL environment failed; inference device does not affect the checkpoint or protocol.
- Full-image subset check: first 16 RGB test images at native resolution: Stage 2 MSE/MAE `0.000131278 / 0.006518074`; joint `0.000120763 / 0.006220614`; crop-trained official baseline `0.010958159 / 0.080615781`.
- Source configuration: `train_conv_base_rtv.py`, `train_refine_rtv.py`, and `train_joint_rtv.py` all point by default to the RGB dataset. `rtv_training_common.py` defines 3-channel base and refine networks; saved base/joint state dictionaries also have 3-channel input weights. The training invocation's actual `--data-root` value is not logged.
- Dataset generation configuration: `run_generate_rtv_voc2012.m` creates Y targets; `run_generate_rtv_voc2012_rgb.m` creates RGB targets. Both specify `lambda=0.01`, `sigma=3.0`, `sharpness=0.02`, `maxIter=4`, split seed `20261008`, and ratios 80/10/10.
- No text training history files were found. Stage 1 and Stage 2 validation values come from `best_loss` in their full checkpoints; the official baseline validation value comes from `best_val` in its checkpoint. The test comparison above was run directly and is documented here, but no separate machine-readable result file was saved.

## Reproduction

The three training entry points, with their current defaults, are:

```bash
python train_conv_base_rtv.py
python train_refine_rtv.py
python train_joint_rtv.py
```

All default to the RGB dataset and 80 epochs. Stage 2 loads `models/conv_base_rtv/conv_base_rtv_best.pth`; Stage 3 loads both Stage 1 and Stage 2 best checkpoints. For a deliberate Y-channel run, configure the model constructors for one channel as well as passing `--data-root`; passing only the Y data path to the current 3-channel models will fail on channel dimensions.

The five completed checkpoints now have a shared evaluation-only command: `python eval_rtv_models.py --device cuda:0 --output-dir eval_results/all_test`. It uses the same RGB test pairs, full-image even-border trimming, `[0,1]` BGR metric scale, and synchronized inference timing for each model. The previously recorded all-test comparison remains an ad hoc result without its original machine-readable output; run the new evaluator to produce reproducible per-image results. The historical crop-trained official baseline is deliberately excluded from the new default comparison.

## Design and Decision Context

- The present task is supervised operator imitation: an ordinary VOC2012 image is paired with its fixed RTV output. RTV output is the target, not an independent clean ground truth.
- Training uses train and validation splits; the test split is held out by the dataset layout. The source directory is VOC2012 trainval, so this is an internal split of those 11,540 images, not the official VOC2012 test set.
- The implementation uses three stages: ConvGuided base, frozen-base three-stage refine, and joint fine-tuning. The joint script optimizes both networks and saves a best pair of weights, but does not write a resumable full checkpoint.
- The historical official baseline was trained on random 256x256 crops and therefore is not resolution-aligned with the three-stage models' full-image validation. New baseline runs default to full-image data collation and inference; they will share the RGB data/splits and image resolution with the three-stage experiment while retaining the official model's distinct architecture and objective.
- The MATLAB generator can produce WLS, L0, rolling-guidance, Gaussian, and bilateral targets, but no paired checkpoints or evaluations for those operators were found here. Traditional operator outputs can be added as direct filter baselines only after fixing their parameters and applying them to the same test inputs; their outputs will define different targets, so compare both fidelity to each operator and visual behavior rather than mixing target metrics.
- A separate literature review follows below. It is background and is not evidence that this local experiment reproduced any published method.

## Pitfalls and Constraints

- README prose currently says the scripts default to the Y dataset, while the actual script constants point to RGB. The code and saved model establish a 3-channel configuration; the exact dataset path passed to the historical stage-training invocation remains unresolved. Update README separately if it is intended as the user-facing run guide.
- Test MSE/MAE measure reproduction of the generated RTV label, not perceptual quality against human ground truth.
- Do not claim Stage 3 completed 80 epochs merely because its CLI default is 80 or because a best checkpoint exists. The current code does not save its epoch or metric.
- Do not conflate the two generated datasets. Both exist, but the default training run uses RGB. Y data cannot be loaded by the current 3-channel model without changing model channels.
- The RGB target is generated by applying the operator independently to RGB channels, as specified by the MATLAB generator. It is a distinct target definition from applying RTV after conversion to Y.

## Open Questions

1. How many epochs did the joint stage actually run, and what was its best validation MSE? No artifact currently records this.
2. What qualitative differences appear between the aligned official baseline and three-stage joint predictions on representative images? The quantitative test split favors the official baseline for RTV target fidelity, but this is not a human-ground-truth perceptual-quality metric.
3. Should the intended main experiment remain RGB, or should it instead be a one-channel Y model? Current code/artifacts indicate RGB, despite the conflicting README prose.
4. What are qualitative results on representative images, including failure cases and edge/texture preservation behavior?

## Next Actions

1. Run `python eval_rtv_models.py --device cuda:0 --output-dir eval_results/all_test` on all 1,154 full-image RGB test pairs. Its `summary.csv`, per-image CSVs, and `protocol.json` provide the first machine-readable shared comparison of Stage 1/2/3, official RTV, and NAFNet patch-256.
2. Keep the aligned full-image baseline separate from the historical crop run; training is complete and its full-test metrics are recorded above.
3. Preserve the generated CSVs and PNGs from the shared evaluator; the older ad hoc comparison did not save its per-image output.
4. Inspect representative predictions from both learned models against inputs and RTV labels; assess texture suppression and edge preservation, not only pixel error.
5. Recover the joint-stage epoch and validation score from a retained log if one exists outside this directory; otherwise mark them unavailable rather than reconstructing them from the checkpoint timestamp.
6. Optionally evaluate classical WLS/RTV/L0/rolling-guidance outputs on the same test images with explicitly fixed parameters. Keep each operator's fidelity score separate because each defines its own target.

## Literature Review Scope

This note records papers whose proposed model is explicitly intended to remove/smooth texture, extract structure, or perform structure-texture decomposition. The main distinction is methodological:

- **Learning-based:** a trainable model is fitted on a collection of images and then applied to new images.
- **Deep-prior optimization:** a neural network is used as an image prior, but parameters are optimized for each input image; this is not a conventionally trained, generalizable model.
- **Mathematical/traditional:** filtering or variational optimization with no learned parameters and no training set.

The word “deep” alone is not evidence that a paper is learning-based. Venue alone is also not evidence.

## Literature Review Status

The paper list below has been corrected and grouped by method type. Dataset counts are reported only when supported by the paper abstract, official code, or an explicit public project description. A missing count is marked **Unresolved**, rather than inferred from a test set.

### A. Learning-based models

#### A.1 A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing

- **Authors:** Qingnan Fan et al.
- **Venue:** ICCV 2017
- **DOI:** `10.1109/ICCV.2017.351`
- **Task relevance:** The paper explicitly applies one deep pipeline to reflection removal and image smoothing. Image smoothing is described as removing insignificant texture while preserving edge structure, so this is directly relevant to Texture Removal.
- **Model type:** Trainable cascaded CNN that estimates edges and reconstructs the image. This is a genuine learning-based model, not a per-image deep prior.
- **Training data:** The abstract states that the reflection-removal branch uses a novel synthetic data generation method providing weak supervision. The exact image count and the exact source images used for the image-smoothing branch are **Unresolved** from the verified metadata collected here; do not quote a count until the full paper or released training code is checked.
- **Important distinction:** The earlier shorthand “Deep Image Smoothing, ICCV 2017” was inaccurate. The correct title is the one above.

#### A.2 Deep Flexible Structure Preserving Image Smoothing

- **Authors:** Mingjia Li et al.
- **Venue:** ACM Multimedia 2022
- **DOI:** `10.1145/3503161.3547857`
- **Task relevance:** Explicitly targets structure-preserving image smoothing, i.e. suppressing insignificant texture while retaining salient structures. The released model can control smoothing strength at inference time.
- **Model type:** Trainable UNet-in-UNet model with an adjuster and smoother. It is a genuine trainable model; inference uses a fixed checkpoint rather than optimizing each test image.
- **Training data (verified from the official repository README):**
  - BSDS500 training set.
  - The first **10,000 MS-COCO training images**, selected in ascending alphabetical filename order.
  - The repository says the released checkpoints are trained with these sources.
  - The official code uses COCO `train2017` for training and `val2017` for validation; the README does not state the exact number of validation images used by the final report.
- **Paired data:** The paper/README state that the smoothing model is trained **without paired data**. Therefore the 10,000 COCO images should not be described as paired input/ground-truth texture-removal examples.
- **Public implementation:** `https://github.com/lime-j/DeepFSPIS`
- **Evaluation artifacts:** The README links results on BSDS500 validation/test sets, but does not provide a single unified benchmark count in the project page.

#### A.3 Deep Scale-Aware Image Smoothing

- **Venue:** ICASSP 2022
- **DOI:** `10.1109/ICASSP43922.2022.9747210`
- **Task relevance:** The abstract explicitly defines image smoothing as smoothing insignificant textures while preserving meaningful structures, and proposes a deep-learning scale-aware model.
- **Model type:** Trainable downscaling-upscaling network with attention. This is a genuine learning-based method.
- **Training data and count:** **Unresolved.** The verified abstract does not state the dataset or number of images. Do not assume BSDS500, COCO, or MIT-Adobe FiveK without checking the paper's experimental section or author code.

#### A.4 Deep Texture and Structure Aware Filtering Network for Image Smoothing

- **Venue:** ECCV 2018 proceedings/lecture-note paper (Springer DOI `10.1007/978-3-030-01225-0_14`).
- **Task relevance:** The title explicitly targets texture- and structure-aware filtering for image smoothing.
- **Model type:** Learning-based filtering network, subject to final verification of the training protocol from the full paper.
- **Training data and count:** **Unresolved.** The bibliographic record and abstract metadata collected here do not expose the dataset size. This paper should remain a candidate until its full experimental section is checked.

### B. Deep-prior optimization, not conventional learning

#### B.1 Structure-Texture Image Decomposition Using Deep Variational Priors

- **Authors:** Youngjung Kim, Bumsub Ham, Minh N. Do.
- **Venue:** IEEE Transactions on Image Processing, 2019 (published online 2018).
- **DOI:** `10.1109/TIP.2018.2889531`
- **Task relevance:** Directly performs structure-texture decomposition and reports texture removal/smoothing applications.
- **Model type:** Deep variational prior optimized with ADMM for each image. The abstract explicitly says the deep prior is learned **“without explicit training data.”** CNNs act as a prior replacing a TV prior; they are not trained once on a dataset and then deployed as a fixed universal model.
- **Dataset:** No conventional training dataset. Experimental images are per-image optimization/evaluation inputs. The number of such images is not a training-set size and should not be reported as one.
- **Decision:** Exclude this paper from the conventional “learning-based model” list, but retain it as a deep-prior/optimization reference.

### C. Mathematical/traditional methods

These methods do not train a network and do not require a training dataset:

1. **Guided Image Filtering**, ECCV 2010 — local linear edge-aware filter.
2. **Domain Transform for Edge-Aware Image and Video Processing**, SIGGRAPH 2011 — recursive/domain-transform filtering.
3. **Image Smoothing via L0 Gradient Minimization**, SIGGRAPH Asia 2011 — gradient-sparsity optimization.
4. **Structure Extraction from Texture via Relative Total Variation**, SIGGRAPH Asia 2012 — RTV variational optimization.
5. **Rolling Guidance Filter**, ECCV 2014 — iterative multi-scale filtering.
6. **A Shape-Aware Structure-Preserving Texture Smoothing Algorithm**, ICIP 2019 — traditional shape/structure-aware smoothing.
7. **Structure-Preserving Texture Smoothing via Scale-Aware Bilateral Total Variation**, IEEE TCSVT 2023 — scale-aware bilateral/TV optimization.

These are valuable baselines, but should not be described as learning-based Texture Removal models.

## Literature Review Artifacts

- OpenAlex record for Deep Variational Priors: its abstract states “without explicit training data” and describes ADMM/deep priors, CNNs replacing a TV prior, and texture-removal applications.
- Official DeepFSPIS repository README: `https://github.com/lime-j/DeepFSPIS`
  - Released checkpoints: BSDS500 train set plus first 10,000 MS-COCO training images in ascending filename order.
  - Training commands use COCO `train2017` and `val2017`.
  - README states training is without paired data.
- OpenAlex/Crossref bibliographic records verified the DOI and venues for the papers listed above.

## Literature Review Reproduction

For the one method with a public implementation and an explicitly documented dataset recipe:

```text
git clone https://github.com/lime-j/DeepFSPIS
```

The repository README gives the training workflow:

```text
python train_adjuster.py --train_dir=<COCO training image dir> --edge_dir=<COCO_edge> --workdir=./train_adjuster
python train_smoother.py --train_dataset_path=<COCO train image dir> --val_dataset_path=<COCO val image dir> --workdir=./train_smoother
```

The released checkpoint recipe is the authoritative count currently available: the standard BSDS500 train split (200 images) plus 10,000 COCO training images, i.e. 10,200 source images in total. This is a count of source images, not paired texture/structure targets.

## Literature Review Decision Context

The most important correction is that **Deep Variational Priors is not a standard learning-based model**. “Deep” describes the prior parameterization, not a trained model with a reusable checkpoint. Its optimization is image-specific.

The direct learning-based candidates for a Texture Removal study are therefore:

1. **A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing** (ICCV 2017).
2. **Deep Flexible Structure Preserving Image Smoothing** (ACM MM 2022).
3. **Deep Scale-Aware Image Smoothing** (ICASSP 2022).
4. **Deep Texture and Structure Aware Filtering Network for Image Smoothing** (ECCV 2018 proceedings), pending full-data verification.

The strongest verified dataset statement among these is currently DeepFSPIS. Its model learns from unpaired natural images and edge maps rather than paired texture-free targets. This is materially different from a supervised decomposition model trained on `(textured image, clean structure image)` pairs.

## Literature Review Pitfalls and Constraints

- Do not call BSDS500, COCO, Brodatz, DTD, or CUReT a dedicated Texture Removal benchmark. They are source/evaluation data used by particular methods or texture databases.
- Do not treat a paper's test image count as a training-set size.
- Do not infer a paired dataset just because a model outputs a smoothed image.
- “Without paired data” does not mean “without training data.” DeepFSPIS is trained on unpaired natural images.
- “Without explicit training data” in Deep Variational Priors does mean that the method is not conventional dataset-trained learning.
- The earlier label “Deep Image Smoothing, ICCV 2017” must be replaced by the exact ICCV title: **A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing**.
- For Deep Scale-Aware Image Smoothing and the 2018 deep texture/structure filtering paper, dataset counts remain unresolved until the full PDFs or official code are inspected.

## Literature Review Open Questions

1. What exact training and validation counts are reported in the ICCV 2017 paper for the image-smoothing branch?
2. What dataset and image count are used by Deep Scale-Aware Image Smoothing (ICASSP 2022)?
3. Does the ECCV 2018 proceedings paper use a trainable dataset-trained network, and what are its exact train/test splits?
4. What exact BSDS500 split count and COCO validation count does the DeepFSPIS paper use in its reported tables?
5. Do these learning-based papers use texture-free ground truth, algorithm-generated smoothing targets, or unpaired/self-supervised losses?

## Literature Review Next Actions

1. Obtain the full PDFs for ICCV 2017, ICASSP 2022, and the ECCV 2018 proceedings paper.
2. Extract the dataset paragraphs and tables verbatim, recording source page/section numbers.
3. Verify whether each model is trained once or optimized per image.
4. Build a compact comparison table with columns: `paper`, `venue`, `model type`, `training data`, `count`, `paired?`, `test data`, `texture-removal evidence`.
5. Keep Deep Variational Priors in a separate “deep-prior optimization” subsection rather than the learning-based subsection.
