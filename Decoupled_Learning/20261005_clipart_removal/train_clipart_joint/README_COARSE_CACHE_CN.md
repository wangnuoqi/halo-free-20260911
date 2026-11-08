# 冻结 base 的预推理缓存训练

在原 exp2-a/exp2-b 命令上增加 `--cached-coarse --augment`。
先单独运行 `clipart_coarse_cache.py`，用对应 exp1 最佳 checkpoint 对固定 train/val list 做预推理；重复预生成只补缺失缓存。Refine 的 `--cached-coarse` 只读取已生成数据并校验 manifest，不运行 base；缺失缓存会报错。
缓存位于数据目录 `openclipart_5000/coarse_cache/exp1-a_q30/{train,val}` 或 `openclipart_5000/coarse_cache/exp1-b_q30/{train,val}`，训练入口与单独预生成入口使用同一默认位置。
每张 coarse 保存 CHW float32 `.npy`，不做 PNG/JPEG 量化。
manifest 记录 base SHA256、质量、通道、数据根目录与划分列表 SHA256；更换这些条件需要新缓存目录。
原始 JPEG/clean 文件应保持不变；若重新生成源图，也应使用新的缓存目录。

训练时只读 JPEG、coarse、clean，不再运行 base，也不再生成低分辨率输入。
三者同步使用随机 0/90/180/270 度旋转及水平翻转；不插值、不再次 JPEG 编码。
验证集无增强。测试集不进入训练/缓存生成流程。
增强的是已有三元组；base 非严格旋转等变，因此增强缓存不等价于对增强输入重新运行 base。
单通道沿用已训练 base 的 PIL `L` 转换；RGB 沿用 RGB 转换。

默认总 batch 8、workers 16，Refine 学习率和 loss 沿用 exp2。
建议缓存增强实验输出用 `ckpt/exp2-a_refine_cached_aug`、`ckpt/exp2-b_refine_cached_aug`，便于保留原实验。
`--resume` 可加载同一路 base 的旧 Refine last checkpoint；恢复后增强策略改变，应作为新的训练阶段记录。
缓存生成脚本：`clipart_coarse_cache.py --channels 1` 或 `--channels 3`，可单独预生成。
当前验证为 CPU 小样本缓存、重复跳过、对齐增强、两通道版本 Refine 前向反向；全量 GPU 缓存由 Windows wintrain 运行。
