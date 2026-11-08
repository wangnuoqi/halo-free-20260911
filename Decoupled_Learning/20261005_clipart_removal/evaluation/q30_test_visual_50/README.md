# Q30 test visual comparison: 50 matched images

These are 50 deterministic, seed-42 random IDs from
`openclipart_5000/test_list.txt`. Each ID has one JPEG input, one original
clean PNG ground truth, and seven saved model predictions. All files are in
this directory so sorting by filename groups each input/GT/prediction set.

Filename pattern: `<test_id>__<variant>.<extension>`.

| Variant | Source |
| --- | --- |
| `input.jpg` | `openclipart_5000/jpeg_pairs_512/jpeg_q30` |
| `GT.png` | `openclipart_5000/jpeg_pairs_512/clean_png` |
| `ARCNN.png`, `FBCNN.png`, `SwinIR.png`, `VIVNet.png` | `baselines/predictions/q30p160` |
| `exp3b_original.png` | `baselines/predictions/q30p160/exp3_b` |
| `exp3b_joint64.png` | `evaluation/q30_exp3b_patch128_noaug_bs64_test/exp3_b` |
| `exp3a_Y.png` | `evaluation/q30_non_swin_timed/exp3_a` (earlier Y-channel experiment) |

`per_image_metrics_50.csv` joins the original evaluation PSNR and MSE by
test ID. Those scores were calculated from model outputs before PNG display
quantization, so remeasuring the saved PNGs may give slightly different
numbers. The earlier `exp3a_Y` run is included for visual context; its
training protocol differs from the protocol160 RGB baselines.
