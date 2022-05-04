import torch
import numpy as np
from imagenet_c import corrupt

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
    
import kornia
import matplotlib.pyplot as plt
from data_model import load_cifar10_data
for x, y in load_cifar10_data(False, batch_size=1):
    x, y = x.cuda(), y.cuda()
    plt.imshow(kornia.tensor_to_image(x[0]))
    plt.savefig('orginal_x.png')
    for name in ok_ops:
        advx = mixupCorruptions(x, name)
        plt.imshow(kornia.tensor_to_image(advx[0]))
        plt.savefig('advx_{}.png'.format(name))
    break