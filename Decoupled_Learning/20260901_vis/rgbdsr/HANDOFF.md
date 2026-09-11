# RGBDSR Visualization

- 已在 `models/` 复制独立的 x8 ConvGuided、DKN、DCTNet 和 Exp1 原始 RF checkpoint，共 4 个文件；来源和输入协议记录在 `models/model_manifest.json`。
- 推理入口：`inference/run_rgbdsr_visualization.py`；一键脚本：`run_rgbdsr_visualization.sh`。
- 输出同时包含四联深度总图、三联毫米绝对误差总图、梯度总图及其 `individual/` 单独子图；子图在同一对应 panel 文件夹内按文件名后缀区分模型；不保存中间数组。
- 默认处理 NYUv2 `benchmark_test_449` 的 x8 数据；`--ids 0 17 42` 可只生成指定样本，省略则处理全部 449 张。
- 已验证：脚本编译通过，4 个复制 checkpoint 均可加载；未在本机执行 449 张完整推理。
