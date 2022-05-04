
import kornia
from torch import clip
from rot_trans_attck import unif # 或者换一个... 
def hsv_attack(x, y, net, num_trials = 50):
    '''
    效果反正很差啦hhhh 因为对h这边没有限制啥的, 
    '''
    img_hsv = kornia.color.rgb_to_hsv(x)
    for i in range(num_trials):
        X_adv_hsv = img_hsv.detach().clone()
        d_h = unif((1,1), 0, 1)
        d_s = unif((1,1), -1, 1) * float(i) / num_trials
        X_adv_hsv[0, 0, :, :] = (img_hsv[0, 0, :, :]+ d_h[0]) % 1.0
        X_adv_hsv[0, 1, :, :] = clip(img_hsv[0, 1, :, :] + d_s[0], 0., 1.)
        X = kornia.color.hsv_to_rgb(X_adv_hsv)
        X = clip(X, 0., 1.)
        out = net(X)
        # print(out)
        if out.argmax(dim=1) != y:
            print("Early stop")
            return X, out
    return X, out
    
'''
from time import time
from data_model import load_cifar10_data, load_cifar10_stander_model
loader = load_cifar10_data(train=False, batch_size=1)
net = load_cifar10_stander_model().cuda().eval()
success = 0
n = 0
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    plt.imshow(kornia.tensor_to_image(x[0]))
    plt.savefig('x.png')
    print(y.cpu().detach().item())
    t1 = time()
    advx, out = hsv_attack(x, y, net)
    t2 = time()
    plt.imshow(kornia.tensor_to_image(advx[0]))
    plt.savefig('advx.png')
    print("time cost ", t2-t1)
    print("out:",out)
    break

'''