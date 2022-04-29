import torch
import numpy as np
from imagenet_c import corrupt

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def CommonCorruptionsAttack(x, y, model, magnitude, name):
    
    x = x.cuda()
    y = y.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).istem() == 0:
        return adv, None
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)

    x_np = x.permute((0,2,3,1)).cpu().numpy()  # We make a copy to avoid changing things in-place
    x_np = (x_np * 255).astype(np.uint8)[:,:,::-1]
    
    for batch_idx, x_np_b in enumerate(x_np):
        corrupt_x = corrupt(x_np_b, corruption_name=name, severity=int(magnitude))
        corrupt_x = corrupt_x.astype(np.float32) / 255.
        adv[ind_non_suc[batch_idx]] = torch.from_numpy(corrupt_x).permute((2,0,1)).cuda()
    return adv, None

def ApplyCorruptions(x, magnitude, name):
    adv = x.clone()
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    x_np = x.permute((0,2,3,1)).cpu().numpy()  
    x_np = (x_np * 255).astype(np.uint8)[:,:,::-1]
    for batch_idx, x_np_b in enumerate(x_np):
        corrupt_x = corrupt(x_np_b, corruption_name=name, severity=int(magnitude))
        corrupt_x = corrupt_x.astype(np.float32) / 255.
        print(corrupt_x.shape)
        adv[batch_idx] = torch.from_numpy(corrupt_x).permute((2,0,1)).cuda()
    return adv

def GaussianNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'gaussian_noise')

def ContrastAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'contrast')

def GaussianBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'gaussian_blur')

def SaturateAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'saturate')

def ElasticTransformAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'elastic_transform')

def JpegCompressionAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'jpeg_compression')

def ShotNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'shot_noise')

def ImpulseNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'impulse_noise')

def DefocusBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'defocus_blur')

def GlassBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'glass_blur')

def MotionBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'motion_blur')

def ZoomBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'zoom_blur')

def FogAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'fog')

def BrightnessAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'brightness')

def PixelateAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'pixelate')

def SpeckleNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'speckle_noise')

def SpatterAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'spatter')