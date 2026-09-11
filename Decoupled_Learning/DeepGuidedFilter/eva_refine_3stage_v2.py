import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class GuidedRefineBlock(nn.Module):
    def __init__(self, base_channels=32, feat_channels=32, a_scale=0.2, b_scale=0.2, use_feat=True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.feat_channels = feat_channels
        self.use_feat = use_feat

        in_ch = 3 + (feat_channels if use_feat else 0)

        self.feat_net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        self.ab_head = nn.Conv2d(feat_channels, 2, 1, bias=True)

    def forward(self, x_raw, x_coarse_k, feat_prev=None):
        g, _, _ = compute_sobel_grad(x_raw, normalize=True)

        if self.use_feat:
            if feat_prev is None:
                feat_prev = torch.zeros(
                    x_raw.size(0), self.feat_channels, x_raw.size(2), x_raw.size(3),
                    device=x_raw.device, dtype=x_raw.dtype
                )
            inp = torch.cat([x_raw, x_coarse_k, g, feat_prev], dim=1)
        else:
            inp = torch.cat([x_raw, x_coarse_k, g], dim=1)

        feat = self.feat_net(inp)
        ab = self.ab_head(feat)
        a_raw = ab[:, 0:1]
        b_raw = ab[:, 1:2]

        a = self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b, feat

class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, feat_channels=32, a_scale=0.2, b_scale=0.2, use_feat=True):
        super().__init__()
        self.stage1 = GuidedRefineBlock(
            base_channels=base_channels,
            feat_channels=feat_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_feat=use_feat,
        )
        self.stage2 = GuidedRefineBlock(
            base_channels=base_channels,
            feat_channels=feat_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_feat=use_feat,
        )
        self.stage3 = GuidedRefineBlock(
            base_channels=base_channels,
            feat_channels=feat_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_feat=use_feat,
        )

    def forward(self, x_raw, x_coarse0):
        a1, b1, feat1 = self.stage1(x_raw, x_coarse0, None)
        delta1 = a1 * x_raw + b1
        x_coarse1 = x_coarse0 + delta1

        a2, b2, feat2 = self.stage2(x_raw, x_coarse1, feat1)
        delta2 = a2 * x_raw + b2
        x_coarse2 = x_coarse1 + delta2

        a3, b3, feat3 = self.stage3(x_raw, x_coarse2, feat2)
        delta3 = a3 * x_raw + b3
        x_final = x_coarse2 + delta3

        return {
            "delta1": delta1, "coarse1": x_coarse1,
            "delta2": delta2, "coarse2": x_coarse2,
            "delta3": delta3, "final":  x_final,
            "a1": a1, "b1": b1,
            "a2": a2, "b2": b2,
            "a3": a3, "b3": b3,
            "feat1": feat1, "feat2": feat2, "feat3": feat3,
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/test_rf_3stage_input'
input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testlabel'

model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/models_retrain/rf_3stage_best_t0.8_rf_v2.pth"

output_dir = "/mnt/d/nuoqi/S2R-HDR/data/test_SIG17/rf3stage_output_v2"
os.makedirs(output_dir, exist_ok=True)

model = CascadeRefine3Stage(base_channels=32, feat_channels=32, a_scale=0.2, b_scale=0.2, use_feat=True)
model = nn.DataParallel(model).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model.eval()

def save_float32_tiff(img, save_path):
    img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)

def crop_to_even(tensor):
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]

def compute_sobel_grad(x, eps=1e-6, normalize=False):
    sobel_x = torch.tensor([[[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1., -2., -1.],
                              [ 0.,  0.,  0.],
                              [ 1.,  2.,  1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g  = torch.sqrt(gx*gx + gy*gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2,3), keepdim=True) + eps)
    return g, gx, gy

class GuidedFilterDataset(Dataset):
    def __init__(self, origin_input_dir, input_dir, label_dir):
        self.origin_input_files = sorted([os.path.join(origin_input_dir, f) for f in os.listdir(origin_input_dir)])
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"INPUT: {len(self.input_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.input_files), len(self.label_files))

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        origin_input = cv2.imread(self.origin_input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input = cv2.imread(self.input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input_hr_h, input_hr_w = input.shape

        filename = os.path.splitext(os.path.basename(input_path))[0]

        return {
            'origin_input_hr': torch.from_numpy(origin_input).unsqueeze(0).unsqueeze(0),
            'input_hr': torch.from_numpy(input).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'filename': filename,
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])

    batch_input = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_input_origin   = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    filenames = []

    for idx, item in enumerate(batch):
        input_origin_crop = item['origin_input_hr'][:, :, :min_h_hr, :min_w_hr]
        input_crop   = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        batch_input_origin[idx] = input_origin_crop
        batch_input[idx] = input_crop
        batch_target_images[idx] = label_crop
        filenames.append(item['filename'])

    return batch_input_origin, batch_input, batch_target_images, filenames

test_dataset = GuidedFilterDataset(
    origin_input_dir = origin_input_test,
    input_dir = input_folder_test,
    label_dir = input_folder_test_label
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

test_mse = 0
test_psnr = 0
total_time = 0
total_images = 0
with torch.no_grad():
    for test_input_origin, test_input, test_label, filenames in tqdm(test_loader, desc="Testing"):
        test_input_origin = crop_to_even(test_input_origin)
        test_input = crop_to_even(test_input)
        test_label = crop_to_even(test_label)

        test_input_origin, test_input, test_label = test_input_origin.to(device), test_input.to(device), test_label.to(device)

        test_input_origin = test_input_origin - 0.5
        test_input = test_input - 0.5
        test_label = test_label - 0.5

        start_time = time.time()
        outputs = model(test_input_origin, test_input)
        end_time = time.time()

        test_output = outputs["final"] + 0.5
        test_output = torch.clamp(test_output, 0, 1)
        test_label  = test_label + 0.5

        batch_time = end_time - start_time
        total_time += batch_time
        total_images += test_input.size(0)

        # for i, fname in enumerate(filenames):
        #     save_float32_tiff(test_output[i], f"{output_dir}/{fname}.tiff")

        psnr, mse = calc_psnr(test_output.detach(), test_label)
        test_mse += mse
        test_psnr += psnr

    avg_test_mse = test_mse / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_infer_time = total_time / total_images

print(f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, Average MSE: {avg_test_mse:.6f}")
print(f"Average inference time per image: {avg_infer_time * 1000:.2f} ms")
