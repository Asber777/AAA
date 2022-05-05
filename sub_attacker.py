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
def op(x, angle, txs1, txs2):
    rots = angle * 0.01745327778 
    theta = ch.tensor([
        [math.cos(rots),math.sin(-rots),txs1],
        [math.sin(rots),math.cos(rots),txs2]
    ], dtype=ch.float).cuda()
    grid = F.affine_grid(theta.unsqueeze(0), x.size())
    X = F.grid_sample(x, grid, align_corners=True)
    return X
        
def random_rt_attack(x, y, net, nb_iter=10, max_rot=15., max_trans=4./32):
    for i in range(nb_iter):
        angle = unif(1, -max_rot, max_rot)
        trans = unif(2, -max_trans, max_trans)
        advx = op(x, angle, trans[0], trans[1])
        if net(advx).argmax(dim=1) != y:
            return advx
    return advx

def grid_rt_attack(x, y, net, max_rot=15., max_trans=4./32):
    limits, granularity = [max_trans, max_trans, max_rot], [5, 5, 10]
    mask = 1 - one_hot(y[0], num_classes=10)
    grid = product(*list(np.linspace(-l, l, num=g) for l, g in zip(limits, granularity)))
    max_logits, best_para, max_out = -1, [None, None, None], None
    for tx, ty, angle in grid:
        output = op(x, angle, tx, ty)
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

def flow_uv(x, flow_layer):
    img_yuv = kornia.color.rgb_to_yuv(x)
    img_y = img_yuv[:, :1, :, :]
    img_uv = img_yuv[:, 1:, :, :]
    flowed_img = flow_layer(img_uv)
    flowed_img = torch.cat([img_y, flowed_img], dim=-3)
    return kornia.color.yuv_to_rgb(flowed_img)

def flow_h(x, flow_layer):
    img_hsv = kornia.color.rgb_to_hsv(x)
    img_h = img_hsv[:, :1, :, :]
    img_sv = img_hsv[:, 1:, :, :]
    flowed_img = flow_layer(img_h)
    flowed_img = torch.cat([flowed_img, img_sv], dim=-3)
    return kornia.color.hsv_to_rgb(flowed_img)

def flow_ab(x, flow_layer):
    img_lab = kornia.color.rgb_to_lab(x)
    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    flowed_img = flow_layer(img_ab)
    flowed_img = torch.cat([img_l, flowed_img], dim=-3)
    return kornia.color.lab_to_rgb(flowed_img)


import matplotlib.pyplot as plt
def flow_perturb(x, y, net, nb_iter=10, tau=50, domain='rgb'):
    match = {'lab': flow_ab, 'hsv': flow_h, 'yuv': flow_uv} # RGB的先放着吧..
    flow_channel = match[domain]
    flow = Flow(x.shape[-2], x.shape[-1]).cuda()
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
    for n in range(nb_iter):
        flowed_img = flow_channel(x, flow)
        flowed_img = clamp(flowed_img, min=0, max=1)
        out = net(flowed_img)
        adv_loss = Adv_loss(out, y, -1)
        flow_loss = tv(flow._pre_flow_field)
        loss = (adv_loss + tau * flow_loss).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return flowed_img

class Flow(torch.nn.Module):
    def __init__(self, height=32, width=32, init_std=0.01):
        super().__init__()
        self.H = height
        self.W = width
        self.basegrid = torch.nn.Parameter(
                torch.cartesian_prod(
                    torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)
                ).view(self.H, self.W, 2).unsqueeze(0)
        )
        self._pre_flow_field = torch.nn.Parameter(
            torch.randn([1, self.H, self.W, 2]) * init_std,
            requires_grad=True,
        )

    def _normalize_grid(self, in_grid):
        grid_x = in_grid[..., 0] # 得到在x方向上的grid偏移量
        grid_y = in_grid[..., 1] # 得到在y方向上的grid偏移量
        # 对每个方向的偏移图分别normalize一下再拼接成原来对size
        return torch.stack([grid_x * 2 / self.H, grid_y * 2 / self.W], dim=-1)

    def forward(self, x):
        grid = self.basegrid + self._normalize_grid(self._pre_flow_field)
        return tf.grid_sample(x, grid, align_corners=True, padding_mode="reflection")

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

from random import choice
class SubAttacker():
    def __init__(self, nb_iter) -> None:
        self.nb_iter = nb_iter
        self.corrupt_list = ['defocus_blur', 'shot_noise', 'saturate', 'gaussian_blur','gaussian_noise', 'jpeg_compression', 'speckle_noise', 'spatter','brightness','zoom_blur', 'motion_blur','contrast']
        self.oplist = [
            lambda x, y, net: flow_perturb(x, y, net, nb_iter, domain='lab'),
            lambda x, y, net: flow_perturb(x, y, net, nb_iter, domain='hsv'),
            lambda x, y, net: flow_perturb(x, y, net, nb_iter, domain='yuv'),
            lambda x, y, net: pgd(x, y, net, nb_iter),
            lambda x, y, net: random_rt_attack(x, y, net, nb_iter),
            lambda x, y, net: grid_rt_attack(x, y, net),
            lambda x, y, net: mixupCorruptions(x, choice(self.corrupt_list))
        ]
        self.num_op = len(self.oplist)
        self.op_name = [
            'lab_flow_attack',
            'hsv_flow_attack',
            'yuv_flow_attack',
            'pgd', 
            'random_rotaion_trans',
            'grid_rotaion_trans',
            'mixupCorruptions'
        ]

    def attack(self, action: int, x, y, net):
        assert 0<= action <self.num_op
        x = x.clone().detach()
        y = y.clone().detach()
        net.eval()
        return self.oplist[action](x, y, net)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from data_model import load_cifar10_data, load_cifar10_stander_model
    net = load_cifar10_stander_model().cuda().eval()
    attack = SubAttacker(10)
    for x, y in load_cifar10_data(batch_size=1):
        x, y = x.cuda(), y.cuda()
        plt.imshow(kornia.tensor_to_image(x[0]))
        plt.savefig('x.png')
        for i in range(attack.num_op):
            print(attack.op_name[i])
            advx = attack.attack(i, x, y, net)
            plt.imshow(kornia.tensor_to_image(advx[0]))
            plt.savefig('advx{}.png'.format(i))
        break
        

# if __name__=='__main__':
#     from data_model import load_cifar10_data, load_cifar10_stander_model
#     import matplotlib.pyplot as plt
#     net = load_cifar10_stander_model().cuda().eval()
#     for x, y in load_cifar10_data(batch_size=1):
#         x, y = x.cuda(), y.cuda()
#         plt.imshow(kornia.tensor_to_image(x[0]))
#         plt.savefig('x.png')
#         advx = flow_perturb(x, y, net, nb_iter=10, domain='rgb')
#         plt.imshow(kornia.tensor_to_image(advx[0]))
#         plt.savefig('advx.png')
#         break

