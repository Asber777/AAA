import math
from matplotlib.colors import hsv_to_rgb
import torch as ch
from itertools import product
from torch.nn.functional import one_hot
import numpy as np
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import kornia
'''
self.max_rot = float(30)
self.max_trans = 3/28.
granularity = [5, 5, 30]
时间花费... 很长...  所以1w个我只看了前331的acc: 292/362 =80.44%
然而这就话费了21分钟之久... 
'''
# reference: https://blog.csdn.net/bbbeoy/article/details/108338487

def unif(size, mini, maxi):
    args = {"from": mini, "to":maxi}
    return ch.FloatTensor(size).uniform_(**args).cuda()

from torch.nn import functional as F
class Spatial:
    def __init__(self, spatial_constraint=15,):
        self.max_rot = float(spatial_constraint)
        self.max_trans = 3/28.

    # x: [1, 3, w, h]
    def op(self, x, angle, txs1, txs2):
        rots = angle * 0.01745327778 
        theta = ch.tensor([
            [math.cos(rots),math.sin(-rots),txs1],
            [math.sin(rots),math.cos(rots),txs2]
        ], dtype=ch.float).cuda()
        grid = F.affine_grid(theta.unsqueeze(0), x.size())
        output = F.grid_sample(x, grid, align_corners=True)
        return output
        
    def random_attack(self, x):
        angle = unif(1, -self.max_rot, self.max_rot)
        trans = unif(2, -self.max_trans, self.max_trans)
        return self.op(x, angle, trans[0], trans[1])

    def grid_attack(self, x, net, y):
        limits, granularity = [self.max_trans, self.max_trans, self.max_rot], [5, 5, 15]
        mask = 1 - one_hot(y[0], num_classes=10)
        grid = product(*list(np.linspace(-l, l, num=g) for l, g in zip(limits, granularity)))
        max_logits = -1
        best_para = [None, None, None]
        max_out = None
        for tx, ty, angle in grid:
            output = self.op(x, angle, tx, ty)
            logits = net(output)
            if logits.argmax(dim=1) != y:
                return output, [angle, tx, ty], True
            logits = softmax(logits, dim=1)
            if (logits*mask).max() > max_logits:
                max_logits = (logits*mask).max()
                best_para = [angle, tx, ty]
                max_out = output
        return max_out, best_para, False # False means fail to attack


'''
from time import time
from tqdm import tqdm
from data_model import load_cifar10_data, load_cifar10_stander_model
loader = load_cifar10_data(train=False, batch_size=1)
attack = Spatial()
net = load_cifar10_stander_model().cuda().eval()
success = 0
n = 0
for x, y in tqdm(loader):
    x, y = x.cuda(), y.cuda()
    # plt.imshow(kornia.tensor_to_image(x[0]))
    # plt.savefig('original_x.png')
    # t1 = time()
    out, best, acc = attack.grid_attack(x, net, y)
    n += 1
    if acc: success += 1
    # t2 = time()
    # print("best parameter is rot:{}, tx:{}, ty:{}, cost {} s".\
    #     format(best[0], best[1], best[2], t2-t1))
    # plt.imshow(kornia.tensor_to_image(out[0]))
    # plt.savefig('x.png')
    # print("original logits is {}, best output logits is {}".format(net(x).detach().cpu().numpy(),\
    #     net(out).detach().cpu().numpy()))
print(success/n)
'''
