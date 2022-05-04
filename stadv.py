import torch.nn.functional as tf
import torch
'''
论文: https://arxiv.org/pdf/2108.02502.pdf
代码实现: https://github.com/ayberkydn/stadv-torch
亲测: 0.9460136217948718 acc (model-standard, cifar10-test)
'''
class Flow(torch.nn.Module):
    def __init__(
        self, height, width, in_batch_size=None, init_std=0.01, param=None,
    ):

        # parameterization is the function to apply
        # to self.flow_field before applying the flow

        super().__init__()

        self.H = height
        self.W = width

        if in_batch_size == None:
            in_batch_size = 1

        # torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)是网格
        # torch.cartesian_prod 生成笛卡尔积, 即坐标, 是一个一纬向量, view成网格
        # unsqueenze是为了后面repeat_interleave broadcast成batch形式的
        self.basegrid = torch.nn.Parameter(
            (
                torch.cartesian_prod(
                    torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)
                )
                .view(self.H, self.W, 2)
                .unsqueeze(0)
                .repeat_interleave(in_batch_size, dim=0)
            )
        )

        if param == None:
            self.parameterization = torch.nn.Identity()
        else:
            self.parameterization = param

        self._pre_flow_field = torch.nn.Parameter(
            torch.randn([in_batch_size, self.H, self.W, 2]) * init_std,
            requires_grad=True,
        )

    def _normalize_grid(self, in_grid):

        """
            Normalize x and y coords of in_grid into range -1, 1 to keep torch.grid_sample happy

        """
        grid_x = in_grid[..., 0] # 得到在x方向上的grid偏移量
        grid_y = in_grid[..., 1] # 得到在y方向上的grid偏移量
        # 对每个方向的偏移图分别normalize一下再拼接成原来对size
        return torch.stack([grid_x * 2 / self.H, grid_y * 2 / self.W], dim=-1)

    def forward(self, x):
        grid = self.basegrid + self._normalize_grid(
            self.parameterization(self._pre_flow_field)
        )

        return tf.grid_sample(x, grid, align_corners=True, padding_mode="reflection")

    def get_applied_flow(self):
        return self.parameterization(self._pre_flow_field)


from piqa.tv import tv
# piqa是一个评估图像质量的库 包含FSIM GMSD MS-GMSD LPIPS PSNR SSIM TV MS-SSIM
# Total Variation (TV) 是衡量整幅图的横纵偏移程度的, 越大代表偏移的越多
# get_applied_flow 是通过flow_layer当前的f求导shift后的图片的. 
def Flow_loss(flow_layer:Flow, keep):
    return tv(flow_layer.get_applied_flow()[keep])


def Adv_loss(adv_logits, target_class, kappa):
    """
    adv_logits: [N]
    target_class: tensor
    kappa: non-positive float
    """
    # 这个就是CWloss的margin loss部分了
    top_two_logits, top_two_classes = torch.topk(adv_logits, 2)
    target_class_logit = adv_logits[range(len(adv_logits)), target_class]

    # 因为传入的在前面判断过, 是pred==target的 所以可以直接计算, 不用判断. 
    '''
    # if top_two_classes[0] == target_class[0]:
    #     nontarget_max = top_two_logits[1]
    # else:
    #     nontarget_max = top_two_logits[0]
    '''
    nontarget_max = top_two_logits[..., 1]
    
    loss =  torch.maximum(target_class_logit - nontarget_max, torch.tensor(kappa))
    return loss


import torch
import matplotlib.pyplot as plt
import numpy as np
import kornia

# 用于可视化flow
def visualize_flow(flow_layer, image=None, grid=False, figsize=[15, 15]):
    H = flow_layer.H
    W = flow_layer.W
    with torch.no_grad():
        flow = flow_layer.get_applied_flow().cpu().numpy()
    if image is None:
        image = np.ones(shape=[flow_layer.H, flow_layer.W, 3])
    plt.figure(figsize=figsize)

    if grid:
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, W, 1))
        ax.set_yticks(np.arange(-0.5, H, 1))
        ax.set_xticklabels(np.arange(1, H + 2, 1))
        ax.set_yticklabels(np.arange(1, W + 2, 1))
        plt.grid()
        plt.imshow(image)

    plt.quiver(
        flow[1], flow[0], units="xy", angles="xy", scale=1,
    )
    plt.show()


def flow_uv(image, flow_layer):
    img_yuv = kornia.color.rgb_to_yuv(image)
    img_y = img_yuv[:, :1, :, :]
    img_uv = img_yuv[:, 1:, :, :]
    flowed_img = flow_layer(img_uv)
    flowed_img = torch.cat([img_y, flowed_img], dim=-3)
    return kornia.color.yuv_to_rgb(flowed_img)


def flow_h(image, flow_layer):
    img_hsv = kornia.color.rgb_to_hsv(image)
    img_h = img_hsv[:, :1, :, :]
    img_sv = img_hsv[:, 1:, :, :]
    flowed_img = flow_layer(img_h)
    flowed_img = torch.cat([flowed_img, img_sv], dim=-3)
    return kornia.color.hsv_to_rgb(flowed_img)


def flow_ab(image, flow_layer):
    img_lab = kornia.color.rgb_to_lab(image)
    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    flowed_img = flow_layer(img_ab)
    flowed_img = torch.cat([img_l, flowed_img], dim=-3)
    return kornia.color.lab_to_rgb(flowed_img)

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]
# TODO: 不知道为什么一直存在继续攻击, 成功的会变失败, 使用索引感觉会隔离梯度的啊, 但是

def experience_on_sift():
    import torch as t 
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, ):
            super(MLP, self).__init__()
            self.fc = t.nn.Linear(10, 3)
        def forward(self, x):
            
            return self.fc(x)
    x = t.randn([10, 10])
    y = t.randint(3, [10])
    y_one_hot=nn.functional.one_hot(y, num_classes=3).float()
    delta = t.randn([10, 10], requires_grad=True)

    net = MLP()
    net.eval()
    optimizer = t.optim.Adam([delta], lr=0.01)
    out = net(x+delta)
    loss = nn.MSELoss(reduction='sum')(out, y_one_hot)

    print(delta.grad)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(delta.grad)

    keep = t.tensor([1, 2, 3])
    out = net(x[keep]+delta[keep])
    loss = nn.MSELoss(reduction='sum')(out, y_one_hot[keep])
    print(delta.grad)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(delta.grad)
    # 证明了keep外的并没有grad



if __name__ == '__main__':
    from data_model import load_cifar10_data, load_cifar10_stander_model
    import tqdm
    tau = 50
    method = 'hsv' # 'yuv' 'lab'
    def max_contrast(img):
        img = img - img.min()
        img = img / img.max()
        return img
        
    net = load_cifar10_stander_model().cuda().eval()
    success_num = 0
    num = 0
    bs = 128
    for img, y in load_cifar10_data(False, batch_size=bs):
        img, target_class = img.cuda(), y.cuda()

        H, W = 32, 32
        if method == 'hsv':
            flow_layer = Flow(H, W, len(img), param=None).to("cuda")
            optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.01)
            flow_channel = flow_h
        elif method == 'yuv':
            flow_layer = Flow(H, W, len(img), param=None).to("cuda")
            optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.01)
            flow_channel = flow_uv
        elif method == 'lab':
            flow_layer = Flow(H, W, len(img), param=None).to("cuda")
            optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.01)
            flow_channel = flow_ab
        pbar = tqdm.tqdm(range(50), desc='t')
        result = img.clone().detach()
        for n in pbar:
            # 找到原因了 是因为flowed_img不断在变化 所以结果也在变 (是这样吗?) 
            flowed_img = flow_channel(img, flow_layer)  
            out = net(flowed_img)
            pred = predict_from_logits(out)
            keep = pred == target_class
            succ = pred != target_class 
            result[succ] = flowed_img[succ]
            if keep.any():
                adv_loss = Adv_loss(out[keep], target_class[keep], -1)
                flow_loss = Flow_loss(flow_layer, keep)
                loss = (adv_loss + tau * flow_loss).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.detach().clone().cpu().item(),
                    keep=keep.clone().detach().sum().cpu().item(),)
            else:
                print("all attacked in {} step".format(n+1))
                break

        out = net(result)
        pred = predict_from_logits(out)
        success_num += (pred != target_class).sum().cpu().item()
        num += bs

    print("acc", success_num/num)

        # # eval all
        # out = net(result)
        # pred = predict_from_logits(out)
        # acc = (pred != target_class).sum().cpu().item()/len(pred)
        # print("acc for all is ", acc)
        # plt.imshow(kornia.tensor_to_image(img[0]))
        # plt.savefig('new_origin.png')
        # plt.imshow(kornia.tensor_to_image(abs(flowed_img[0])))
        # plt.savefig('new_flowed_img.png')
        # plt.imshow(kornia.tensor_to_image(max_contrast(flowed_img[0] - img[0])))
        # plt.savefig('new_max_contrast.png')

        # # # src.utils.visualize_flow(flow_layer, kornia.tensor_to_image(flowed_img))
        # diff_img = flowed_img - img
        # print(f"Mean: {diff_img.mean()}")
        # print(f"Max: {diff_img.max()}")
        # print(f"Min: {diff_img.min()}")
        # break