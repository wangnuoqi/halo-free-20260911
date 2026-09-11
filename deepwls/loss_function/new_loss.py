"""
@公式：(out-in)^2 + λ(Wh*(out_gxR^2+G^2+B^2)+Wv*(out_gyR^2+G^2+B^2))
Wh=（|in_gxR|^α+|in_gxG|^α+|in_gxB|^α+exp）

"""

import torch.nn as nn
import torch

from util.tool import img_gradient, weight


class new_loss(nn.Module):
    """
    x输入：图像
    """
    def __init__(self):  # __init__初始化对象值
        super(new_loss, self).__init__()  # super继承父类的方法和属性nn.Module

    def forward(self, x, y, lamd, alpha, target=None):
        """
        x: 输入 图像
        y: 输出
        """
        batchsz = x.size()[0]
        channel = 1
        h_x = x.size()[2]  # height
        w_x = x.size()[3]
        image = x[:, 0:3, ...]

        omigaH, omigaV = weight(image, batchsz, h_x, w_x, lamd, alpha) 

        reference = image if target is None else target[:, 0:3, ...]
        loss1 = torch.mean(torch.pow((reference - y), 2))

        y_grad_x, y_grad_y = img_gradient(y, 2)
        
        # (y_i partial horizon)
        y_grad_xR = y_grad_x[:, 0, ...].reshape(batchsz, channel, h_x, w_x)
        y_grad_xG = y_grad_x[:, 1, ...].reshape(batchsz, channel, h_x, w_x)
        y_grad_xB = y_grad_x[:, 2, ...].reshape(batchsz, channel, h_x, w_x)

        y_grad_yR = y_grad_y[:, 0, ...].reshape(batchsz, channel, h_x, w_x)
        y_grad_yG = y_grad_y[:, 1, ...].reshape(batchsz, channel, h_x, w_x)
        y_grad_yB = y_grad_y[:, 2, ...].reshape(batchsz, channel, h_x, w_x)


        L2 = torch.mul(omigaH, y_grad_xR+y_grad_xG+y_grad_xB)
        L3 = torch.mul(omigaV, y_grad_yR+y_grad_yG+y_grad_yB)
        loss2 = torch.mean(torch.add(L2, L3))
        # loss2 = torch.mean(torch.mul(lamd, torch.add(L2, L3)))
        return loss1 + loss2


# test = new_loss()
