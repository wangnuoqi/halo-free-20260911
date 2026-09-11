# Decoupled-params 训练脚本总结

这个目录里的训练代码大体可以分成两条路线：

1. `ConvGuidedFilter` 路线：把 `(lambda, alpha)` 作为条件输入，让网络学习从低分辨率输入预测滤波结果，再用高分辨率引导图恢复到原图分辨率。
2. `WLS/Refine` 路线：直接学习 WLS 风格迭代求解器，或者把已有 coarse 结果当成初值，再做多阶段细化。

这份 README 的目标不是解释每一行实现，而是总结每个 `train*.py` 的训练策略、它和其它脚本的关系，以及更适合拿来做什么实验。

## 共性设置

大多数脚本共享下面这些训练约定：

- 数据来自 `S2R-HDR` 下的参数化数据集，`lambda` 和 `alpha` 通过文件名里的 `-l...-a...-h` 规则解析出来。
- `collate_fn` 会把一个 batch 内的图裁到共同的最小尺寸，避免可变分辨率带来的拼接问题。
- 高分辨率输入和测试时通常会做 `crop_to_even`，避免后续下采样或插值时出现奇数边长问题。
- 评价指标基本统一为 `MSE + PSNR`，最佳模型按验证集 `MSE` 保存。
- `ConvGuidedFilter` 系列和 `train_wls_unrolled.py` 都用 `MSE + 梯度损失` 的组合损失，梯度项通过 `scale = loss_grad / loss_mse` 做了自适应重标定，避免梯度损失量纲过大。
- 多数脚本默认训练 `80` 个 epoch，优化器是 `Adam`，学习率调度是 `StepLR(step_size=30, gamma=0.5)`。

## 脚本总览

| 脚本 | 核心思路 | 训练策略关键词 |
| --- | --- | --- |
| `train_convguided_base.py` | 从零训练参数条件化的 `DeepGuidedFilter` | 全参数训练、late injection、baseline |
| `train_convguided_base_paramdebug.py` | 为参数条件化做可观测性更强的 debug 版 | 全参数训练、通道更大、打印条件编码统计 |
| `train_convguided_base_paraminput_freezeb.py` | 直接把参数图拼到输入里，并从单参数模型迁移 | 预训练迁移、冻结大部分 backbone、只调部分层 |
| `train_convguided_base_paramtransfer_freezefront.py` | 把单参数模型拆前后半，参数调制放到后半段 | 预训练迁移、冻结前半段、训练后半段和调制器 |
| `train_convguided_base_paramadapter.py` | 在预训练 backbone 中插一个轻量 adapter 学参数变化 | 预训练迁移、冻结 backbone/gf、只训 adapter+tail |
| `train_rf_3stagev3_falsenorm.py` | 以 coarse 结果为起点做三级细化 | 级联 refine、边缘区域监督、逐 stage 学残差 |
| `train_wls_unrolled.py` | 把 WLS 求解写成可学习的多步迭代 | unrolled optimization、显式各向异性权重更新 |

## 1. `train_convguided_base.py`

这是最标准的参数化 `ConvGuidedFilter` 基线。

- 结构上由两部分组成：
  - `LRNet`：在低分辨率输入上预测一个低分辨率滤波结果。
  - `ConvGuidedFilter`：把低分辨率预测和高分辨率引导图结合，恢复到高分辨率输出。
- 参数注入方式是 `late injection`：
  - 先提取图像特征。
  - 再用 `(lambda, alpha)` 经过两层全连接生成 `InstanceNorm` 的 weight/bias。
  - 最后在注入卷积后的特征上逐样本做条件归一化。
- 训练是完全从零开始的，没有显式加载预训练权重。
- 损失是：
  - `MSE`
  - `Sobel gradient loss`
  - 最终形式是 `loss_mse + loss_g / scale_g`

这版最适合做主 baseline，因为它没有冻结策略，也没有迁移学习假设，能直接反映“参数条件化的 guided filter 网络”本身能学到什么。

### 结果文件路径

- 完整 checkpoint:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_mix_param_deeplr_injectlate.pth`
- 最后一轮权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_last_mix_param_deeplr_injectlate.pth`
- 验证集最优权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_best_mix_param_deeplr_injectlate.pth`
- 续训默认读取:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_mix_param_deeplr_injectlate.pth`
- TensorBoard 日志目录:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir/convguided_filter_mix_param_deeplr_injectlate_<timestamp>/`

## 2. `train_convguided_base_paramdebug.py`

这版本质上是 `base` 的 debug/诊断变体，不是另一个完全不同的方法。

- 仍然是从零训练的 `DeepGuidedFilter`。
- 条件化方式还是基于参数生成 `InstanceNorm` 的 weight/bias。
- 与 `base` 的主要区别：
  - `BASE_CHANNELS=64`，网络容量更大。
  - `LRNet` 更直接，使用空洞卷积串联而不是 `ResidualDilatedBlock` 形式。
  - 每隔一段 iteration 会额外打印：
    - 原始参数范围
    - 归一化后的参数范围
    - 条件编码产生的 weight/bias 统计量
    - 跨样本条件编码差异

这版更像是“验证参数是否真的被网络用起来”的实验脚本。它适合做排查和消融，不适合作为最终推荐方案。

### 结果文件路径

- 完整 checkpoint:
  `/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_full_mix_paramdebug64.pth`
- 最后一轮权重:
  `/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_last_mix_paramdebug64.pth`
- 验证集最优权重:
  `/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_best_mix_paramdebug64.pth`
- 续训默认读取:
  `/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_full_mix_paramdebug64.pth`
- TensorBoard 日志目录:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir/convguided_filter_mix_paramdebug64_<timestamp>/`

注意：这一版的模型文件默认不写在 `Decoupled-params/model/`，而是写到旧目录 `Decoupled_Learning/HDRv1/models_retrain/`。

## 3. `train_convguided_base_paraminput_freezeb.py`

这版改了两个核心点：参数输入形式 + 训练策略。

### 结构变化

- 参数不再通过单独的 late injection 注入。
- 做法是先把 `(lambda, alpha)` 归一化，再扩展成两个常数通道，和 `x_lr` 直接拼接，形成 `x_lr_cond` 输入 `lr` 子网。
- 也就是说，这版是“input-level conditioning”，而不是“feature-level modulation”。

### 训练策略

- 先加载一个单参数预训练模型 `PRETRAINED_PATH`。
- 然后启用 `freeze_version_b`：
  - 整个 `gf` 冻结。
  - `lr` 默认全部冻结。
  - 只重新打开 `lr` 的头部层和尾部层，以及最终输出层。
  - 中间大段 backbone 保持冻结。

这代表它不是要重学整个网络，而是想在保留单参数模型主体能力的前提下，只靠少量层适配到双参数条件输入。

适合的实验问题是：  
“如果直接把参数当输入拼进去，并且尽量少改预训练网络，是否已经足够？”

### 结果文件路径

- 完整 checkpoint:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_mix_paraminput_freezeb.pth`
- 最后一轮权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_last_mix_paraminput_freezeb.pth`
- 验证集最优权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_best_mix_paraminput_freezeb.pth`
- 续训默认读取:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_mix_paraminput_freezeb.pth`
- TensorBoard 日志目录:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir/convguided_filter_mix_<timestamp>/`

## 4. `train_convguided_base_paramtransfer_freezefront.py`

这版是更明确的“前后半拆分迁移”方案。

- 先把单参数 `lr` 网络拆成三段：
  - `lr_front`
  - `lr_back`
  - `lr_out`
- 参数调制器 `LateParamModulator` 插在 `lr_back` 之后、`lr_out` 之前。
- 同样会先加载单参数预训练权重，但不是整体硬搬，而是把旧 `lr` 权重按索引映射到 `front/back/out` 三部分。

默认冻结策略是：

- `lr_front` 冻结
- `gf` 冻结
- `lr_back` 可训练
- `param_mod` 可训练
- `lr_out` 可训练
- 可选参数 `--unfreeze-gf-conv-a` 可以只把 guided filter 里的 `conv_a` 解冻

它背后的假设很清楚：

- 前半部分主要负责提取比较通用的图像表征，可以继承单参数模型。
- 参数相关的变化，更适合在网络后半段或输出前局部调整。

相比 `paraminput_freezeb`，这版更强调“保留前端、改造后端”。

### 结果文件路径

- 完整 checkpoint:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_mix_paramtransfer_freezefront.pth`
- 最后一轮权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_last_mix_paramtransfer_freezefront.pth`
- 验证集最优权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_best_mix_paramtransfer_freezefront.pth`
- 续训默认读取:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_mix_paramtransfer_freezefront.pth`
- TensorBoard 日志目录:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir/convguided_filter_mix_paramtransfer_freezefront_<timestamp>/`

## 5. `train_convguided_base_paramadapter.py`

这版是几种迁移策略里最“轻量”的一个。

- 仍然从单参数预训练 `ConvGuidedFilter` 出发。
- 先把原来的 `lr` 子网拆成：
  - `lr_backbone`
  - `lr_tail`
- 中间插入一个很小的 `ParamAdapter`：
  - `(lambda, alpha)` 经过 MLP 生成 `gamma/beta`
  - 用 `feat * (1 + gamma) + beta` 做特征仿射调制
- 参数先做归一化，再整体减去 `0.5`，让调制中心靠近零

冻结策略更激进：

- `lr_backbone` 冻结
- `gf` 冻结
- `lr_tail` 可训练
- `adapter` 可训练

这版的目标不是最大化自由度，而是测试：  
“在一个已经学会滤波的 backbone 上，只插一个很小的参数适配器，能不能低成本适配到多参数设置？”

如果你要做参数高效迁移，这版是最像 PEFT 思路的。

### 结果文件路径

- 完整 checkpoint:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_adapter_param.pth`
- 最后一轮权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_last_adapter_param.pth`
- 验证集最优权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_best_adapter_param.pth`
- 续训默认读取:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_full_adapter_param.pth`
- TensorBoard 日志目录:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir/convguided_filter_adapter_param_<timestamp>/`

## 6. `train_rf_3stagev3_falsenorm.py`

这版不再是 guided filter 直接回归，而是三级 cascade refine。

### 输入和输出关系

- 输入不是 `(x_lr, x_hr)` 这种 low/high-res 配对。
- 它直接读：
  - `origin_input_hr`：原始输入图
  - `input_hr`：已有 coarse 结果
  - `target_img`：目标图
- 模型目标不是从头生成结果，而是逐步修正 coarse 结果。

### 三个 stage 的职责

- Stage 1:
  - `Stage1DeltaNet` 预测一个初始残差 `delta1`
  - 得到 `coarse1 = coarse0 + delta1`
- Stage 2:
  - `ABAdjustNet` 预测逐像素 `a2, b2`
  - 用 `delta2 = a2 * delta1 + b2` 再次修正
- Stage 3:
  - 再做一轮 `a3, b3` 调整
  - 最终得到 `x_final = coarse0 + delta3`

这其实是在学“如何逐步修正已有滤波结果”，而不是直接拟合目标。

### 监督方式

- 不是直接只对 final output 做损失。
- 先构造 `delta_gt = gt - coarse0`。
- 然后分别监督 `delta1 / delta2 / delta3`，最后把三项损失平均。
- 核心损失 `cascade_loss` 使用 `edge_guided_loss`，并默认：
  - `thresh_mode='otsu'`
  - `loss_type='l1'`
  - `normalize_grad=False`

也就是说，这版更关注边缘和结构性误差，尤其强调 coarse 解在难点区域的修正能力。

它适合回答的问题是：  
“已有 WLS 或其它滤波结果时，能否用小网络分阶段把错误逐步修回来？”

### 结果文件路径

- 完整 checkpoint:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/rf_3stage_full_otsu_v3_falsenorm_param.pth`
- 最后一轮权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/rf_3stage_last_otsu_v3_falsenorm_param.pth`
- 验证集最优权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/rf_3stage_best_otsu_v3_falsenorm_param.pth`
- 续训默认读取:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/rf_3stage_full_otsu_v3_falsenorm_param.pth`
- TensorBoard 日志目录:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir/rf_3stage_otsu_v3_falsenorm_<timestamp>/`

## 7. `train_wls_unrolled.py`

这版是目录里最“方法化”的训练脚本，因为它把 WLS 求解过程显式展开成了可学习迭代。

### 结构

- `LowResBaseNet`
  - 和 guided-filter baseline 很像，先在低分辨率上预测一个 base 结果。
  - 同样通过参数条件化影响低分辨率预测。
- `LearnedWLSNet`
  - 先把 `base_lr` 双线性插值成 `base_hr`
  - 再串联多个 `WLSUpdateBlock`
- `WLSUpdateBlock`
  - 根据 `guide / base / current / guide_grad` 预测权重修正项和 source term
  - 显式构造各向异性权重 `w_x / w_y`
  - 显式计算加权拉普拉斯 `lap`
  - 按照 WLS 风格更新式迭代得到下一步 `current`

### 训练策略

- 默认 `--steps 5`，相当于把 5 次迭代 unroll 到网络里。
- 损失不只看最终输出：
  - `final` 对 GT 的 `MSE`
  - `final` 对 GT 的梯度损失
  - `base_hr` 对 GT 的 `MSE`
- 总损失是：
  - `loss_mse + loss_grad / scale_grad + 0.2 * loss_base`

这说明训练目标不仅要求最终多步迭代结果好，还要求初始 base 解本身不能太差，否则整个迭代起点会不稳定。

这版和 `ConvGuidedFilter` 系列最大的区别是：

- `ConvGuidedFilter` 更像学习一个直接映射。
- `train_wls_unrolled.py` 更像学习一个“可解释的优化器”。

### 结果文件路径

- 完整 checkpoint:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/learned_wls_full.pth`
- 最后一轮权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/learned_wls_last.pth`
- 验证集最优权重:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/learned_wls_best.pth`
- 续训默认读取:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/model/learned_wls_full.pth`
- TensorBoard 日志目录:
  `/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir/learned_wls_unrolled_<timestamp>/`

## 如何理解这些脚本之间的关系

如果按研究路线来读，可以把它们理解成下面这个演化过程：

1. `train_convguided_base.py`
   从零开始做参数条件化 guided filter，建立最基本 baseline。
2. `train_convguided_base_paramdebug.py`
   检查参数条件化到底有没有真正起作用，做调试和诊断。
3. `train_convguided_base_paraminput_freezeb.py`
   尝试最直接的条件注入方式，并利用单参数模型做小改动迁移。
4. `train_convguided_base_paramtransfer_freezefront.py`
   更系统地把迁移限制在后半段，测试“冻结前端、改后端”的可行性。
5. `train_convguided_base_paramadapter.py`
   进一步缩小可训练部分，用 adapter 做参数高效适配。
6. `train_rf_3stagev3_falsenorm.py`
   放弃直接回归，改成 coarse-to-fine 三级修正。
7. `train_wls_unrolled.py`
   再往前一步，把 WLS 迭代过程本身写进模型结构里。

## 我对这些脚本训练策略的归纳

如果只看“训练策略”而不是网络细节，这个目录实际上在反复比较四种思路：

- 从零学习一个条件化滤波器
  - `train_convguided_base.py`
  - `train_convguided_base_paramdebug.py`
- 基于单参数模型做迁移学习
  - `train_convguided_base_paraminput_freezeb.py`
  - `train_convguided_base_paramtransfer_freezefront.py`
  - `train_convguided_base_paramadapter.py`
- 对已有 coarse 结果做多阶段残差细化
  - `train_rf_3stagev3_falsenorm.py`
- 直接学习 WLS 风格的迭代求解器
  - `train_wls_unrolled.py`

从实验设计上看，重点并不是单纯比较网络深浅，而是在比较：

- 参数应该在哪里注入最有效
- 单参数预训练模型是否值得迁移
- 迁移时应该解冻哪些层
- 相比直接回归，迭代 refinement 或 unrolled solver 是否更稳

## 建议的阅读顺序

如果你以后还要继续维护这些脚本，我建议按下面顺序读：

1. `train_convguided_base.py`
2. `train_convguided_base_paramdebug.py`
3. `train_convguided_base_paraminput_freezeb.py`
4. `train_convguided_base_paramtransfer_freezefront.py`
5. `train_convguided_base_paramadapter.py`
6. `train_rf_3stagev3_falsenorm.py`
7. `train_wls_unrolled.py`

前五个是一条连续的 guided-filter 参数化路线，后两个是另外两种更偏“优化/细化”的路线。
