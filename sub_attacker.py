'''
下面包括了 
1. mixup的污染攻击corruption
2. hsv_attack
3. 

'''

import torch
import torch as ch
import numpy as np
import kornia
import math
from torch import clip
from imagenet_c import corrupt
from itertools import product
import torch.nn.functional as tf
from torch.nn.functional import one_hot
from torch.nn.functional import softmax

def unif(size, mini, maxi):
    args = {"from": mini, "to":maxi}
    return ch.FloatTensor(size).uniform_(**args).cuda()

ok_ops = ['defocus_blur', 'shot_noise', 'saturate', 'gaussian_blur','gaussian_noise', 'jpeg_compression', 'speckle_noise', 'spatter','brightness','zoom_blur', 'motion_blur','contrast']
def mixupCorruptions(x, name, alpha=0.5, magnitude=1):
    assert name in ok_ops 
    x_np = x[0].permute((1,2,0)).cpu().detach().clone().numpy()
    x_np = (x_np * 255).astype(np.uint8)# [:,::-1] #这个操作会左右翻转
    corrupt_x = corrupt(x_np, corruption_name=name, severity=int(magnitude))
    corrupt_x = corrupt_x.astype(np.float32)
    x_np = x_np.astype(np.float32)
    mixup = (alpha * x_np + (1-alpha) * corrupt_x) / 255.
    mixup = torch.from_numpy(mixup).permute((2,0,1)).cuda()
    return torch.unsqueeze(mixup, 0)

def hsv_attack(x, y, net, num_trials = 50): # 效果很差..
    img_hsv = kornia.color.rgb_to_hsv(x)
    for i in range(num_trials):
        X_adv_hsv = img_hsv.detach().clone()
        d_h = unif((1,1), 0, 1)
        d_s = unif((1,1), -1, 1) * float(i) / num_trials
        X_adv_hsv[0, 0, :, :] = (img_hsv[0, 0, :, :]+ d_h[0]) % 1.0
        X_adv_hsv[0, 1, :, :] = clip(img_hsv[0, 1, :, :] + d_s[0], 0., 1.)
        X = kornia.color.hsv_to_rgb(X_adv_hsv)
        X = clip(X, 0., 1.)
        if net(X).argmax(dim=1) != y:
            return X
    return X

from torch.nn import functional as F
class Spatial:
    def __init__(self, spatial_constraint=15,):
        self.max_rot = float(spatial_constraint)
        self.max_trans = 4/32.

    # x: [1, 3, w, h]
    def op(self, x, angle, txs1, txs2):
        rots = angle * 0.01745327778 
        theta = ch.tensor([
            [math.cos(rots),math.sin(-rots),txs1],
            [math.sin(rots),math.cos(rots),txs2]
        ], dtype=ch.float).cuda()
        grid = F.affine_grid(theta.unsqueeze(0), x.size())
        X = F.grid_sample(x, grid, align_corners=True)
        return X
        
    def random_attack(self, x):
        angle = unif(1, -self.max_rot, self.max_rot)
        trans = unif(2, -self.max_trans, self.max_trans)
        return self.op(x, angle, trans[0], trans[1])

    def grid_attack(self, x, net, y):
        limits, granularity = [self.max_trans, self.max_trans, self.max_rot], [5, 5, 15]
        mask = 1 - one_hot(y[0], num_classes=10)
        grid = product(*list(np.linspace(-l, l, num=g) for l, g in zip(limits, granularity)))
        max_logits, best_para, max_out = -1, [None, None, None], None
        for tx, ty, angle in grid:
            output = self.op(x, angle, tx, ty)
            logits = net(output)
            if logits.argmax(dim=1) != y:
                return output
            logits = softmax(logits, dim=1)
            if (logits*mask).max() > max_logits:
                max_logits = (logits*mask).max()
                best_para = [angle, tx, ty]
                max_out = output
        return max_out

from piqa.tv import tv
def Adv_loss(adv_logits, target_class, kappa):
    top_two_logits, top_two_classes = torch.topk(adv_logits, 2)
    target_class_logit = adv_logits[range(len(adv_logits)), target_class]
    nontarget_max = top_two_logits[..., 1]
    loss =  torch.maximum(target_class_logit - nontarget_max, torch.tensor(kappa))
    return loss

def flow_uv(x, y, flow_layer):
    img_yuv = kornia.color.rgb_to_yuv(x)
    img_y = img_yuv[:, :1, :, :]
    img_uv = img_yuv[:, 1:, :, :]
    flowed_img = flow_layer(img_uv)
    flowed_img = torch.cat([img_y, flowed_img], dim=-3)
    return kornia.color.yuv_to_rgb(flowed_img)

def flow_h(x, y, flow_layer):
    img_hsv = kornia.color.rgb_to_hsv(x)
    img_h = img_hsv[:, :1, :, :]
    img_sv = img_hsv[:, 1:, :, :]
    flowed_img = flow_layer(img_h)
    flowed_img = torch.cat([flowed_img, img_sv], dim=-3)
    return kornia.color.hsv_to_rgb(flowed_img)

def flow_ab(x, y, flow_layer):
    img_lab = kornia.color.rgb_to_lab(x)
    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    flowed_img = flow_layer(img_ab)
    flowed_img = torch.cat([img_l, flowed_img], dim=-3)
    return kornia.color.lab_to_rgb(flowed_img)

def flow_rgb(x, y, flow_layer):
    flowed_img = flow_layer(x)
    return flowed_img

class Flow(torch.nn.Module):
    def __init__(self, net, height=32, width=32, init_std=0.01, tau=50, domain_name='rgb'):
        super().__init__()
        self.net = net
        self.H = height
        self.W = width
        self.tau = tau
        self.basegrid = torch.nn.Parameter(
                torch.cartesian_prod(
                    torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)
                ).view(self.H, self.W, 2).unsqueeze(0)
        )
        self._pre_flow_field = torch.nn.Parameter(
            torch.randn([1, self.H, self.W, 2]) * init_std,
            requires_grad=True,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        match = {'rgb': flow_rgb, 'lab': flow_ab, 'hsv': flow_h, 'yuv': flow_uv}
        self.flow_channel = match[domain_name]

    def _normalize_grid(self, in_grid):
        grid_x = in_grid[..., 0] # 得到在x方向上的grid偏移量
        grid_y = in_grid[..., 1] # 得到在y方向上的grid偏移量
        # 对每个方向的偏移图分别normalize一下再拼接成原来对size
        return torch.stack([grid_x * 2 / self.H, grid_y * 2 / self.W], dim=-1)

    def forward(self, x):
        grid = self.basegrid + self._normalize_grid(self._pre_flow_field)
        return tf.grid_sample(x, grid, align_corners=True, padding_mode="reflection")

    def perturb(self, x, y, num_step=10):
        for n in range(num_step):
            # 找到原因了 是因为flowed_img不断在变化 所以结果也在变 (是这样吗?) 
            flowed_img = self.flow_channel(x, self)
            out = self.net(flowed_img)
            adv_loss = Adv_loss(out, y, -1)
            flow_loss = tv(self._pre_flow_field)
            loss = (adv_loss + self.tau * flow_loss).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return flowed_img

import torch.nn as nn
from torch import clamp
def pgd(x, y, net, nb_iter=10, eps=8./255, eps_iter=2./255, 
        clip_min=0.0, clip_max=1.0, random_init = True):
    x, y = x.detach().clone(), y.detach().clone()
    # init delta randomly
    delta = torch.zeros_like(x)
    delta = nn.Parameter(delta)
    if random_init:
        delta.data.uniform_(-1, 1)
        delta.data *= eps
        delta.data = clamp(x + delta.data, min=clip_min, max=clip_max) - x
    # begin PGD
    delta.requires_grad_()
    for _ in range(nb_iter):
        outputs = net(x + delta)
        loss = nn.CrossEntropyLoss(reduction="sum")(outputs, y)
        loss.backward()
        # first limit delta in [-eps,eps] then limit data in [clip_min,clip_max](inf_ord)
        grad_sign = delta.grad.data.sign()
        grad_sign *= eps_iter
        delta.data = delta.data + grad_sign
        delta.data = clamp(delta.data, -eps, eps)
        delta.data = clamp(x.data + delta.data, clip_min, clip_max
                            ) - x.data
        delta.grad.data.zero_()
    x_adv = clamp(x + delta, clip_min, clip_max)
    return x_adv.data

