# FBCNN OpenClipart RGB 重训练

入口 `train_fbcnn_clipart.py`，直接导入 `FBCNN/models/network_fbcnn.py` 官方网络。
不使用预训练权重；初始化沿用官方 color 配置 orthogonal gain 0.2、BN uniform。
RGB in/out；nc=[64,128,256,512]、nb=4、act_mode=BR、strideconv/convtranspose。
读取固定 train/val list 与已有 jpeg_q30/clean_png，不重新压缩。
protocol160 使用同步随机裁剪 128×128，不做水平翻转、旋转或颜色增强；验证完整512图，无增强。
预测质量控制默认由网络预测，不把真实 q 输入重建分支；Q30 辅助标签为0.7。
损失 L1(image,GT)+0.001*L1(q_pred,q_label)。固定Q30仅评估固定质量复原，不能验证多质量盲估计能力。
Adam lr=2e-5；MultiStepLR [200000,400000,1200000,1600000,1800000] 按optimizer step衰减。
protocol160 使用160 epochs、总batch8、workers16；每epoch每图抽一个patch，
约74400个optimizer steps（训练 loader 丢弃最后的不完整 batch）。学习率在
40/80/120 epoch 衰减，避免直接使用官方百万级 iteration milestone。
160epoch是本地从零训练预算，并非复现论文总训练时长。
四卡DataParallel，每卡2个patch；验证batch1。
输出 baselines/ckpt/fbcnn_rgb_q30/{fbcnn_best.pth,fbcnn_last.pth,history.jsonl,config_used.json}。
best按完整验证集RGB MSE选取，同时记录global PSNR与逐图mean PSNR；test list不参与选模。
续训 --resume <fbcnn_last.pth>，恢复optimizer与scheduler；epochs为总目标轮数。
验证：实际train pair读取、形状/q标签检查、官方模型小尺寸前向反向、语法；尚未启动Windows四卡正式训练。
