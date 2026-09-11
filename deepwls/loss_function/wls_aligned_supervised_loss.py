import torch
import torch.nn as nn


class WLSAlignedSupervisedLoss(nn.Module):
    """Supervised loss with a smoothness term aligned to total-pipeline-v1/wls.m."""

    def __init__(self, small_num=1e-4, log_eps=1e-12):
        super().__init__()
        self.small_num = small_num
        self.log_eps = log_eps

    @staticmethod
    def _to_single_channel(tensor):
        if tensor.size(1) == 1:
            return tensor
        return tensor.mean(dim=1, keepdim=True)

    @staticmethod
    def _forward_diff(tensor):
        diff_x = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        diff_y = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        return diff_x, diff_y

    @staticmethod
    def _pad_like_wls(diff_x, diff_y):
        diff_x = torch.nn.functional.pad(diff_x, (0, 1, 0, 0), mode="constant", value=0.0)
        diff_y = torch.nn.functional.pad(diff_y, (0, 0, 0, 1), mode="constant", value=0.0)
        return diff_x, diff_y

    def forward(self, inputs, outputs, targets, wls_lambda, wls_alpha, smooth_weight=1.0):
        guidance = self._to_single_channel(inputs).clamp_min(self.log_eps)
        guidance = torch.log(guidance)
        output_gray = self._to_single_channel(outputs)

        recon_loss = torch.mean((outputs - targets) ** 2)

        guide_dx, guide_dy = self._forward_diff(guidance)
        out_dx, out_dy = self._forward_diff(output_gray)

        weight_x = wls_lambda / (torch.abs(guide_dx).pow(wls_alpha) + self.small_num)
        weight_y = wls_lambda / (torch.abs(guide_dy).pow(wls_alpha) + self.small_num)

        weight_x, weight_y = self._pad_like_wls(weight_x, weight_y)
        out_dx, out_dy = self._pad_like_wls(out_dx, out_dy)

        smooth_loss = torch.mean(weight_x * (out_dx ** 2) + weight_y * (out_dy ** 2))
        total_loss = recon_loss + smooth_weight * smooth_loss

        return total_loss, recon_loss, smooth_loss
