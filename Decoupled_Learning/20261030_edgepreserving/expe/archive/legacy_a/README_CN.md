# 旧 A1/A2 归档

这里保留旧固定均值统计版本的模型名称和训练产物，不作为当前 A 系列正式训练入口。

| 原路径 | 归档路径 | 状态 |
| --- | --- | --- |
| `expe/runs/a_series/viv_gf_a1` | `a_series/viv_gf_a1` | 双卡每卡16、global32，完成60K |
| `expe/runs/a_series/viv_gf_a2` | `a_series/viv_gf_a2` | 双卡每卡16、global32，完成60K |
| `expe/runs/joint/viv_gf_a1` | `joint_viv_gf_a1` | 旧四卡 DataParallel，只有中间 checkpoint |
| `expe/runs/joint/viv_gf_a2` | `joint_viv_gf_a2` | 旧四卡 DataParallel，只有中间 checkpoint |

`variants.py` 保存这些 checkpoint 的原始变体配置。`evaluate_expc.py` 与 `diagnostics.py` 仍可按 checkpoint 内的旧 `protocol.variant` 加载它们；正式训练注册表不再提供旧 A1/A2。原双卡脚本仅用于追溯，不作为新实验入口。固定单窗口 `viv_gf_a1_box3`、`viv_gf_a1_box9` 仍在正式注册表中。

新实验采用 `viv_gf_a1_stats`、`viv_gf_a2_stats_edge`、`viv_gf_a1_direct`、`viv_gf_a2_direct_edge` 和 `viv_gf_a1_fixed_boxes`。旧 A1 checkpoint 与固定双窗口结构兼容，global batch32，低于 E1 的64，只作为历史参考。当前 A 系列脚本会在新目录按有效 batch64从头重训 `viv_gf_a1_fixed_boxes`，不续训归档 checkpoint。
