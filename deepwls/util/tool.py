import math
import random

import torch


def weight(imgs, batch, height, width, lamd, alpha):
    """
    输入图像，样本，高，宽
    生成高斯权重
    """
    # lamd = 10
    # alpha = 0.08  # 0.002, 0.01, 0.05
    # alpha = produce_apha(batch, height, width)
    # lamd = produce_lamd(batch, height, width)
    alpha = pow(alpha, 2)
    # sigima = 5
    # alpha = 0.5
    
    gradH, gradV = img_gradient(imgs, 2)
    rh = gradH[:, 0, ...].reshape(batch, 1, height, width)  # 保证维度
    gh = gradH[:, 1, ...].reshape(batch, 1, height, width)
    bh = gradH[:, 2, ...].reshape(batch, 1, height, width)
    rv = gradV[:, 0, ...].reshape(batch, 1, height, width)
    gv = gradV[:, 1, ...].reshape(batch, 1, height, width)
    bv = gradV[:, 2, ...].reshape(batch, 1, height, width)
    deno = (-2)*alpha
    # omigaH = sigima*torch.div((rh+gh+bh), sigima)
    # omigaH = torch.pow(omigaH, alpha)

    # omigaV = sigima*torch.div((rv+gv+bv), sigima)
    # omigaV = torch.pow(omigaV, alpha)
    omigaH = rh+gh+bh
    # omigaH2 = produce_wlog(batch, height, width)
    # omigaH = omigaH1+omigaH2
    # sigama = produce_sigama(batch, height, width)
    # sigama = -2*torch.pow(sigama, 2)
    # omigaH = torch.div(omigaH, sigama)
    # omigaH = torch.exp(omigaH)
    
    omigaH = torch.div(omigaH, deno)
    omigaH = torch.exp(omigaH)
    omigaH = torch.mul(lamd, omigaH)
    
    omigaV = rv+gv+bv
    # omigaV2 = produce_wlog(batch, height, width)
    # omigaV = omigaV1+omigaV2
    # omigaV = torch.div(omigaV, sigama)
    # omigaV = torch.exp(omigaV)
    omigaV = torch.div(omigaV, deno)
    omigaV = torch.exp(omigaV)
    omigaV = torch.mul(lamd, omigaV)

    return omigaH, omigaV


def img_grad(x, flag):
    """
    计算x的梯度, 随机取sigma， 计算∂x^2 + ∂y^2的值，若小于sigma则将∂x,∂y置为0, 否则不变
    :param x: [b, ch, height, width]
    :param flag: 1R,2G,3B
    :return:
    """
    batchsz = x.size()[0]
    channel = x.size()[1]
    h_x = x.size()[2]  # height
    w_x = x.size()[3]  # width

    # 计算梯度∂x, ∂y (-1, 1)
    # b*c*256*255 1:包含第一列，：w_x-1 不包含这一列 ，含左不含右
    x_grad_x = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
    x_grad_y = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]  # b*c*255*256
    device = x.device
    x_grad_x = torch.cat((x_grad_x, torch.zeros(
        batchsz, channel, h_x, 1, device=device)), dim=3)  # 32*1*128*128
    x_grad_y = torch.cat((x_grad_y, torch.zeros(
        batchsz, channel, 1, w_x, device=device)), dim=2)  # 32*1*128*128
    # x_grad_x = torch.cat((x_grad_x, x[:, :, :, 0:1] - x[:, :, :, w_x-1:w_x]), dim=3)  # b*c*256*256
    # x_grad_y = torch.cat((x_grad_y, x[:, :, 0:1, :] - x[:, :, h_x-1:h_x, :]), dim=2)  # b*c*256*256
    # 随机设置阈值sigma为0.001-0.5
    thresh = []
    if flag:
        sigma = torch.zeros((batchsz, 1, 1, 1), device=device)
        for i in range(batchsz):
            a = round(random.uniform(0.01, 1.2), 3)
            # a = 0.5
            sigma[i, 0, 0, 0] = a
            thresh.append(a)
        # 计算∂x^2 + ∂y^2, 把各通道求和得到一个单通道图
        # temp_x = torch.sum(torch.pow(x_grad_x, 2), dim=1).unsqueeze(1)
        # temp_y = torch.sum(torch.pow(x_grad_y, 2), dim=1).unsqueeze(1)
        # 计算|∂x| + |∂y|, 把各通道求和得到一个单通道图
        temp_x = torch.sum(torch.abs(x_grad_x), dim=1).unsqueeze(1)
        temp_y = torch.sum(torch.abs(x_grad_y), dim=1).unsqueeze(1)
        # x,y方向求和python
        temp = temp_x + temp_y
        # 单通道图中大于sigma的置为1，小于sigma的置为0
        zeros = torch.zeros((batchsz, 1, 1, 1), device=device)
        ones = torch.ones((batchsz, 1, 1, 1), device=device)
        temp = torch.where(temp < sigma, zeros, ones)
        x_grad_x = torch.mul(x_grad_x, temp)
        x_grad_y = torch.mul(x_grad_y, temp)

        # ths = [0.05, 0.06, 0.07, 0.08, 0.1]
        # for i in range(batchsz):
        #     sigma = round(random.uniform(0.1, 1.5), 1)
        #     # 将x, y方向梯度图各通道求和得到一个单通道图
        #     temp_x = torch.abs(x_grad_x[:, 0, ...]) + torch.abs(x_grad_x[:, 1, ...]) + torch.abs(x_grad_x[:, 2, ...])
        #     temp_y = torch.abs(x_grad_y[:, 0, ...]) + torch.abs(x_grad_y[:, 1, ...]) + torch.abs(x_grad_y[:, 2, ...])
        #     # 单通道图中大于sigma的置为1，小于sigma的置为0
        #     zeros = torch.zeros_like(temp_x)
        #     ones = torch.ones_like(temp_x)
        #     temp_x = torch.where(temp_x < sigma, zeros, ones)
        #     temp_y = torch.where(temp_y < sigma, zeros, ones)
        #     # 用该单通道图与原梯度图点乘得到处理后的梯度图
        #     for i in range(channel):
        #         x_grad_x[:, i, ...] = torch.mul(x_grad_x[:, i, ...], temp_x)
        #         x_grad_y[:, i, ...] = torch.mul(x_grad_y[:, i, ...], temp_y)

        #     thresh.append(sigma)
    # (-1， 1) -> (0, 1)

    x_grad_x = (x_grad_x + 1) * 0.5
    x_grad_y = (x_grad_y + 1) * 0.5
    return x_grad_x, x_grad_y


def img_gradient(x, flag):
    """
    直接计算梯度
    #flag 真：梯度flag次方
    #  假：直接进行梯度绝对值运算
    """
    batchsz = x.size()[0]
    channel = x.size()[1]
    h_x = x.size()[2]  # height
    w_x = x.size()[3]  # width

    # b*c*256*255 1:包含第一列，：w_x-1 不包含这一列 ，含左不含右
    x_grad_x = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
    x_grad_y = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]  # b*c*255*256
    x_grad_x = torch.cat(
        (x_grad_x, x[:, :, :, 0:1] - x[:, :, :, w_x-1:w_x]), dim=3)  # 32*1*128*128
    x_grad_y = torch.cat(
        (x_grad_y, x[:, :, 0:1, :] - x[:, :, h_x-1:h_x, :]), dim=2)  # 32*1*128*128
    x_grad_x = torch.abs(x_grad_x)
    x_grad_y = torch.abs(x_grad_y)

    if flag:
        # 求平方
        x_grad_x = torch.pow(x_grad_x, flag)  # 广播可能会造成异常
        x_grad_y = torch.pow(x_grad_y, flag)
    # else:
    #     x_grad_x = torch.mul(torch.add(x_grad_x, 1), 0.5)
    #     x_grad_y = torch.mul(torch.add(x_grad_y, 1), 0.5)

    return x_grad_x, x_grad_y


def produce_lamd(batch, h, w):
    """
    产生batch个 h*w的值随机的张量
    :param batch:
    :param h:
    :param w:
    :return:
    """
    lamd_batch = torch.zeros((batch, 1, h, w), device=h.device if hasattr(h, "device") else "cuda")

    for i in range(batch):
        
        # lam = random.uniform(1, 3000)
        # lam = round(lam/255, 2)
        lam = round(random.uniform(0.001, 1.001), 3)
        # lam = random.uniform(1, 10)
        # tmp = torch.full((1, 1, h, w), round(pow(10, lam), 5), device="cuda")
        tmp = torch.full((1, 1, h, w), lam, device=lamd_batch.device)
        lamd_batch[i, 0, ...] = tmp

    return lamd_batch


def produce_apha(batch, h, w):
    """
    产生batch个 h*w的值随机为1-10的张量
    :param batch:
    :param h:
    :param w:
    :return:
    """
    apha_batch = torch.zeros((batch, 1, h, w), device=h.device if hasattr(h, "device") else "cuda")
    # lams = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # l2_loss

    for i in range(batch):
        # lamd = int(random.random() * 10 + 1)  # 1 - 10  # l1_loss
        # index = int(random.random() * 11)  # l2_loss
        apha = round(random.uniform(0.001, 0.101), 3)  # 0.2的时候结果从0.001开始就不行了
        tmp = torch.full((1, 1, h, w), apha, device=apha_batch.device)
        apha_batch[i, 0, ...] = tmp

    return apha_batch


def produce_sigama(batch, h, w):
    """
    产生batch个 h*w的值随机为1-10的张量
    :param batch:
    :param h:
    :param w:
    :return:
    """
    apha_batch = torch.zeros((batch, 1, h, w), device=h.device if hasattr(h, "device") else "cuda")
    # lams = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # l2_loss

    for i in range(batch):
        # lamd = int(random.random() * 10 + 1)  # 1 - 10  # l1_loss
        # index = int(random.random() * 11)  # l2_loss
        apha = round(random.uniform(0.008, 0.11), 4)
        tmp = torch.full((1, 1, h, w), apha, device=apha_batch.device)
        apha_batch[i, 0, ...] = tmp

    return apha_batch


def produce_wlog(batch, h, w):
    """
    产生batch个 h*w的值随机的张量
    :param batch:
    :param h:
    :param w:
    :return:
    """
    logw = torch.zeros((batch, 1, h, w), device=h.device if hasattr(h, "device") else "cuda")

    for i in range(batch):
        minw = math.pow(9, (-2)*math.pow(0.008, 2))
        maxw = math.pow(3600, (-2)*math.pow(0.11, 2))
        # lam = round(random.uniform(0.8, 1.5), 4)
        lam = round(random.uniform(minw, maxw), 4)
        log1 = math.log(lam)
        # tmp = torch.full((1, 1, h, w), round(pow(10, lam), 5), device="cuda")
        tmp = torch.full((1, 1, h, w), round(log1), device=logw.device)
        logw[i, 0, ...] = tmp

    return logw


def guiyi(tensors):
    """
    归一化,1个通道的四阶张量
    """
    batch = tensors.size(0)
    h = tensors.size(2)
    w = tensors.size(3)
    guiyi = torch.zeros((batch, 1, h, w), device=tensors.device)

    for i in range(batch):
        cut = tensors[i, 0, ...]
        avg = torch.mean(cut)
        summ = avg*h*w
        tmp = torch.full((1, 1, h, w), summ, device=tensors.device)
        guiyi[i, 0, ...] = tmp
    
    norm = torch.div(tensors, guiyi)
    
    return norm



def compute_grad(x, th):
    """
    MyDataset需要用的一个函数
    :param x: [ch, height, width]
    """
    channel = x.size()[0]
    h_x = x.size()[1]  # height
    w_x = x.size()[2]  # width

    # 计算梯度 取绝对值
    x_grad_x = torch.abs(x[:, :, 1:] - x[:, :, :w_x - 1])  # 128*127
    x_grad_y = torch.abs(x[:, 1:, :] - x[:, :h_x - 1, :])  # 127*128

    # 修改size
    x_grad_x = torch.cat((x_grad_x, torch.zeros(
        channel, h_x, 1)), dim=2)  # 32*1*128*128
    x_grad_y = torch.cat((x_grad_y, torch.zeros(
        channel, 1, w_x)), dim=1)  # 32*1*128*128

    if th:
        m = torch.nn.Threshold(th, 0)  # if x < 0.5 => 0
        x_grad_x = m(x_grad_x)
        x_grad_y = m(x_grad_y)
        del m

    return x_grad_x, x_grad_y

# if __name__ == "__main__":
#     a = torch.randn((2, 3, 3, 3), device="cuda")
#     img_grad(a, 1)
