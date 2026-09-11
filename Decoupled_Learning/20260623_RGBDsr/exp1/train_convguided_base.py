import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

import os
import datetime
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import random
from nyuv2_full_dataset import NYUv2FullImageDataset
from convguided_model import SurveyDGF

DEFAULT_TRAIN_ROOT = Path(r"D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2")
DEFAULT_TEST_ROOT = Path(r"D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449")

parser = argparse.ArgumentParser(description="Train the full-image RGB-guided NYUv2 model.")
parser.add_argument("--scale", type=int, choices=(4, 8, 16), default=4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
args = parser.parse_args()

TRAIN_IMAGE_COUNT = 1000
VALIDATION_COUNT = 100
DEPTH_SCALE_TRAIN = 1000.0  # resplit train .npy values are stored in meters
DEPTH_SCALE_TEST = 1.0      # benchmark test PNG values are stored in millimeters
FULL_IMAGE_BATCH_SIZE = args.batch_size
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

if FULL_IMAGE_BATCH_SIZE < 1:
    raise ValueError("--batch-size must be positive")
if args.epochs < 1:
    raise ValueError("--epochs must be positive")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(args.seed)

##### - 数据加载 - #######
# Legacy image-folder loader below expects already normalized depth inputs.
# The active full-image loader is NYUv2FullImageDataset and does per-image norm.
class GuidedFilterDataset(Dataset):
    def __init__(self, depth_lr_dir, rgb_hr_dir, depth_label_dir, max_depth_value=1.0):
        depth_lr_names = self._list_names(depth_lr_dir)
        rgb_hr_names = self._list_names(rgb_hr_dir)
        depth_label_names = self._list_names(depth_label_dir)
        self.names = sorted(depth_lr_names & rgb_hr_names & depth_label_names)
        if not self.names:
            raise RuntimeError(f"No matching files found in {depth_lr_dir}, {rgb_hr_dir}, {depth_label_dir}")

        self.depth_lr_files = [os.path.join(depth_lr_dir, f) for f in self.names]
        self.rgb_hr_files = [os.path.join(rgb_hr_dir, f) for f in self.names]
        self.depth_label_files = [os.path.join(depth_label_dir, f) for f in self.names]
        self.max_depth_value = max_depth_value
        print(f"Depth LR: {len(self.depth_lr_files)}, RGB HR: {len(self.rgb_hr_files)}, Depth label: {len(self.depth_label_files)}")

    @staticmethod
    def _list_names(folder):
        return {
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        depth_lr = read_depth01(self.depth_lr_files[idx], self.max_depth_value)
        rgb_hr = read_rgb01(self.rgb_hr_files[idx])
        depth_label = read_depth01(self.depth_label_files[idx], self.max_depth_value)

        input_hr_h, input_hr_w = depth_label.shape
        input_lr_h, input_lr_w = depth_lr.shape

        return {
            'depth_lr': torch.from_numpy(depth_lr).unsqueeze(0),
            'rgb_hr': torch.from_numpy(rgb_hr).permute(2, 0, 1),
            'depth_label': torch.from_numpy(depth_label).unsqueeze(0),
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
            'input_height_lr': input_lr_h,
            'input_width_lr': input_lr_w
        }

def read_depth01(path, max_depth_value):
    if cv2 is not None:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read depth image: {path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
    else:
        with Image.open(path) as img:
            depth = np.array(img)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
    return depth.astype('float32') / max_depth_value

def read_rgb01(path):
    if cv2 is not None:
        rgb = cv2.imread(path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read RGB image: {path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        with Image.open(path) as img:
            rgb = np.array(img.convert("RGB"))
    return rgb.astype('float32') / 255.0

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])
    min_h_lr = min([item['input_height_lr'] for item in batch])
    min_w_lr = min([item['input_width_lr'] for item in batch])   

    batch_rgb_hr = torch.zeros(len(batch), 3, min_h_hr, min_w_hr)
    batch_depth_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_depth_label = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        depth_lr_crop = item['depth_lr'][:, :min_h_lr, :min_w_lr]
        rgb_hr_crop = item['rgb_hr'][:, :min_h_hr, :min_w_hr]
        depth_label_crop = item['depth_label'][:, :min_h_hr, :min_w_hr]

        # 放到 batch
        batch_depth_lr[idx] = depth_lr_crop
        batch_rgb_hr[idx] = rgb_hr_crop
        batch_depth_label[idx] = depth_label_crop

    return batch_depth_lr, batch_rgb_hr, batch_depth_label

############ 处理数据附带的部分函数 #############
# ------- crop 文件避免产生奇数影响采样和输出 ---------------#
def crop_to_even(tensor):
    """
    将输入 Tensor 裁剪到高宽为偶数，保持左上角对齐（裁掉底部和右边）。
    tensor: [B, C, H, W]
    返回裁剪后的 tensor
    """
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]
#############  对输入的normalized 处理 #############
def normalize_hdr(img, scale=None, mode='P95'):
    if scale is None:
        if mode == 'P95':
            scale = 1.0 / torch.quantile(img, 0.95)
        elif mode == 'max':
            scale = 1.0 / img.max()
        else:
            raise ValueError("Unsupported mode")
    return img * scale, scale

def _to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W) 或 (B,1,H,W)
    if x.dim() == 3:  # (B,H,W)
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        return x
    # 简单均值转灰度（需要更严格可以改为加权）
    return x.mean(dim=1, keepdim=True)

def _sobel_grad(x: torch.Tensor, eps: float = 1e-6):
    # 反射填充，边缘更稳
    x = F.pad(x, (1,1,1,1), mode='reflect')
    kx = torch.tensor([[[[-1.,0.,1.],
                         [-2.,0.,2.],
                         [-1.,0.,1.]]]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[[[-1.,-2.,-1.],
                         [ 0., 0., 0.],
                         [ 1., 2., 1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    mag = torch.sqrt(gx*gx + gy*gy + eps)
    return gx, gy, mag

def sobel_gradient_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        mode: str = 'vector',   # 'vector' 或 'magnitude'
                        reduction: str = 'mean',
                        eps: float = 1e-6) -> torch.Tensor:
    """
    pred, target: (B,C,H,W) 或 (B,1,H,W)
    mode='vector'  : L1(|∂x| + |∂y|)（推荐，方向+幅值都对齐）
    mode='magnitude': L1(|∇|)        （仅幅值）
    """
    pred_g = _to_gray(pred)
    targ_g = _to_gray(target)

    pgx, pgy, pm = _sobel_grad(pred_g, eps)
    tgx, tgy, tm = _sobel_grad(targ_g, eps)

    if mode == 'vector':
        loss = (pgx - tgx).abs() + (pgy - tgy).abs()
    elif mode == 'magnitude':
        loss = (pm - tm).abs()
    else:
        raise ValueError("mode must be 'vector' or 'magnitude'")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_state(model, state_dict):
    """Load checkpoints saved with either plain modules or DataParallel."""
    model_is_parallel = isinstance(model, torch.nn.DataParallel)
    checkpoint_is_parallel = bool(state_dict) and all(key.startswith("module.") for key in state_dict)
    if model_is_parallel and not checkpoint_is_parallel:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    elif checkpoint_is_parallel and not model_is_parallel:
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)


model = SurveyDGF(num_features=64)
if (device.type == 'cuda'
        and torch.cuda.device_count() > 1
        and FULL_IMAGE_BATCH_SIZE >= torch.cuda.device_count()):
    model = torch.nn.DataParallel(model)
model = model.to(device)

##### 构建 full-image 训练、validation 和最终 test ####
train_dataset = NYUv2FullImageDataset(
    args.train_root,
    split="train",
    scale=args.scale,
    indices=range(0, TRAIN_IMAGE_COUNT - VALIDATION_COUNT),
    depth_unit_scale=DEPTH_SCALE_TRAIN,
    augment=True,
)
validation_dataset = NYUv2FullImageDataset(
    args.train_root,
    split="train",
    scale=args.scale,
    indices=range(TRAIN_IMAGE_COUNT - VALIDATION_COUNT, TRAIN_IMAGE_COUNT),
    depth_unit_scale=DEPTH_SCALE_TRAIN,
    augment=False,
)
test_dataset = NYUv2FullImageDataset(
    args.test_root,
    split="test",
    scale=args.scale,
    depth_unit_scale=DEPTH_SCALE_TEST,
    augment=False,
)

# Full 480x640 images are intentionally trained with batch size 1.
train_loader = DataLoader(train_dataset, batch_size=FULL_IMAGE_BATCH_SIZE, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=False)
validation_loader = DataLoader(validation_dataset, batch_size=FULL_IMAGE_BATCH_SIZE, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=FULL_IMAGE_BATCH_SIZE, shuffle=False, num_workers=0,
                         pin_memory=True, drop_last=False)

criterion = torch.nn.L1Loss(reduction='mean')

##### tensorboard 启动指令 #######
task_name = f"nyuv2_rgb_guided_depth_x{args.scale}"
model_version = f"conv_survey_dgf_x{args.scale}"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
PROJECT_ROOT = Path(__file__).resolve().parent
log_dir = PROJECT_ROOT / "log_dir" / f"{task_name}_{timestamp}"

writer = SummaryWriter(log_dir=log_dir)

# --- 在每个 epoch 结束后保存 ---
def save_checkpoint(epoch, best_loss, is_best=False):
    # 保存完整状态，方便续训
    full_path = os.path.join(save_dir, f"rgb_guided_depth_x{args.scale}_full.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }, full_path)
    print(f"Checkpoint saved: {full_path}")

    # 保存纯模型 (方便推理或部署)
    last_model_path = os.path.join(save_dir, f"rgb_guided_depth_x{args.scale}_last.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, f"rgb_guided_depth_x{args.scale}_best.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Val RMSE cm: {best_loss:.6f}) ***")

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10( 1 / mse)
    return psnr, mse

def calc_mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def evaluate_full_image_loader(loader, crop=6):
    """Evaluate normalized predictions after restoring the per-image depth scale."""
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    normalized_mae = 0.0
    normalized_psnr = 0.0
    with torch.no_grad():
        for depth_lr, rgb_hr, depth_label, depth_min, depth_max, _ in loader:
            depth_lr = depth_lr.to(device)
            rgb_hr = rgb_hr.to(device)
            depth_label = depth_label.to(device)
            prediction = model(depth_lr, rgb_hr).clamp(0.0, 1.0)

            depth_range = (depth_max - depth_min).to(device).view(-1, 1, 1, 1)
            depth_offset = depth_min.to(device).view(-1, 1, 1, 1)
            prediction_raw = prediction * depth_range + depth_offset
            target_raw = depth_label * depth_range + depth_offset
            if crop:
                prediction_raw = prediction_raw[:, :, crop:-crop, crop:-crop]
                target_raw = target_raw[:, :, crop:-crop, crop:-crop]
            diff = prediction_raw - target_raw
            squared_error += torch.sum(diff * diff).item()
            pixel_count += diff.numel()

            normalized_mae += calc_mae(prediction, depth_label)
            normalized_psnr += calc_psnr(prediction, depth_label)[0]

    if pixel_count == 0:
        raise RuntimeError("No pixels were evaluated")
    count = len(loader)
    return {
        "rmse_mm": float(np.sqrt(squared_error / pixel_count)),
        "rmse_cm": float(np.sqrt(squared_error / pixel_count) / 10.0),
        "normalized_mae": normalized_mae / count,
        "normalized_psnr": normalized_psnr / count,
    }

# def get_optimizer_and_scheduler(model, lr_1=0.01, lr_2=0.001, switch_epoch=0, total_epochs=150):
#     optimizer = optim.Adam(model.parameters(), lr=lr_1)

#     def lr_lambda(epoch):
#         # 前 switch_epoch：返回 1.0（保持 lr=lr_1）
#         # 后面：返回 lr_2 / lr_1（缩小到 0.01）
#         return 1.0 if epoch < switch_epoch else lr_2 / lr_1

#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     return optimizer, scheduler

# 初始化
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
best_loss = float("inf")
save_dir = PROJECT_ROOT / "models" / model_version
save_dir.mkdir(parents=True, exist_ok=True)
epochs = args.epochs

###### 重新加载训练模型 ###########
start_epoch = 0
resume_path = os.path.join(save_dir, f"rgb_guided_depth_x{args.scale}_full.pth")
if os.path.exists(resume_path):
    print(f"=> Resuming training from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)

    # 注意：这里的 model 已经是 DataParallel(model).to(device) 了
    load_model_state(model, checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 这里保存的是 epoch+1，所以下一轮就从这个 epoch 开始
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float("inf"))

    print(f"=> Resumed from epoch {start_epoch}, best_loss = {best_loss:.6f}")
else:
    print("=> No checkpoint found, training from scratch.")

debug_mode = True
if __name__ == "__main__":
######## -- 模型训练 -- #############
    seed_everything(42) 
    #for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mae = 0
        epoch_psnr = 0
        for batch_idx, (depth_lr, rgb_hr, depth_label, _, _, _) in enumerate(train_loader):
            # if batch_idx > 0:  # 只跑第一个 iter
            #     break       

            depth_lr, rgb_hr, depth_label = depth_lr.to(device), rgb_hr.to(device), depth_label.to(device)

            # 前向
            output = model(depth_lr, rgb_hr)
            eps = 1e-8
            loss_mae = criterion(output, depth_label)
            loss_g = sobel_gradient_loss(output, depth_label, mode='vector')      # 推荐
            output_lr = F.interpolate(output, size=depth_lr.shape[-2:], mode='area')
            loss_consistency = F.l1_loss(output_lr, depth_lr)
            scale_g = (loss_g / (loss_mae + eps)).detach().clamp_min(eps)
            loss = loss_mae + loss_g / scale_g + loss_consistency

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算 MAE 和 PSNR
            if not torch.isfinite(output).all() or not torch.isfinite(depth_label).all():
                print("Warning: NaN or Inf detected in output or labels!")
                print("Output stats:", output.min().item(), output.max().item(), output.mean().item())
                print("Label stats:", depth_label.min().item(), depth_label.max().item(), depth_label.mean().item())

                # 打印前几个值（方便观察是否过大）
                print("Sample output values:", output.view(-1)[:10])
                print("Sample label values:", depth_label.view(-1)[:10])

            psnr, mse = calc_psnr(output.detach(), depth_label)
            mae = calc_mae(output.detach(), depth_label)
            if batch_idx % 200 == 0:
                print(f"iter MAE loss: {mae:.6f}")
            epoch_mae += mae
            epoch_psnr += psnr
            # 每 50 step 记录到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar('train/mae', mae, global_step)
                writer.add_scalar('train/psnr', psnr, global_step)
                writer.add_scalar('train/mae_loss', loss_mae.item(), global_step)
                writer.add_scalar('train/consistency_l1', loss_consistency.item(), global_step)
            #t4 = time.time()
            #if batch_idx % 20 == 0:
            #    print(f"[Batch {batch_idx}] Data: {t1-t0:.3f}s | Forward: {t2-t1:.3f}s | Backward: {t3-t2:.3f}s | Log: {t4-t3:.3f}s", flush=True)
        #scheduler.step()
        avg_epoch_mae = epoch_mae / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
            # 打印本 epoch 的平均指标
        print(f"Epoch {epoch+1}/{epochs} "
            f"- MAE: {avg_epoch_mae:.6f} "
            f"- PSNR: {avg_epoch_psnr:.2f} dB ")
        writer.add_scalar('train/mae_epoch', avg_epoch_mae, epoch)
        writer.add_scalar('train/psnr_epoch', avg_epoch_psnr, epoch)
        scheduler.step()

        # Validation uses the held-out 100 training images; final test is untouched.
        validation_metrics = evaluate_full_image_loader(validation_loader, crop=6)
        print(f"Epoch {epoch+1}/{epochs} "
            f"- Val MAE(norm): {validation_metrics['normalized_mae']:.6f} "
            f"- Val PSNR(norm): {validation_metrics['normalized_psnr']:.2f} dB "
            f"- Val RMSE: {validation_metrics['rmse_cm']:.4f} cm")
        writer.add_scalar('validation/mae_normalized', validation_metrics['normalized_mae'], epoch)
        writer.add_scalar('validation/psnr_normalized', validation_metrics['normalized_psnr'], epoch)
        writer.add_scalar('validation/rmse_cm', validation_metrics['rmse_cm'], epoch)

        is_best = validation_metrics['rmse_cm'] < best_loss
        if is_best:
            best_loss = validation_metrics['rmse_cm']

        # 保存模型
        save_checkpoint(epoch, validation_metrics['rmse_cm'], is_best=is_best)

    best_model_path = os.path.join(save_dir, f"rgb_guided_depth_x{args.scale}_best.pth")
    if os.path.exists(best_model_path):
        load_model_state(model, torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_full_image_loader(test_loader, crop=6)
    print(f"Final test on 449 images: RMSE={test_metrics['rmse_cm']:.4f} cm "
          f"(normalized MAE={test_metrics['normalized_mae']:.6f})")
    writer.add_scalar('final_test/rmse_cm', test_metrics['rmse_cm'], epochs)

    # 关闭 TensorBoard writer
    writer.close()
