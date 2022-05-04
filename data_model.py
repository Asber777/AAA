import os
import robustbench as rb
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
# from torch.utils.data import Subset
import torchvision.transforms as transforms
# from torch.nn import Softmax
from robustbench.utils import load_model

# define where to stor data and model and RL agent
root = '/home/.faa'
model_path = os.path.join(root, 'model')
data_path = os.path.join(root, 'data')

# load dataloader and model set selected from robustbench for attacking. 

def load_cifar10_data(train=False, batch_size=100):
    cifar10 = datasets.CIFAR10(data_path, train=train, transform=transforms.ToTensor())
    loader = DataLoader(cifar10, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

def load_cifar10_stander_model():
    return load_model('Standard', dataset='cifar10', threat_model='Linf',model_dir=model_path)

LINF_MODEL = ['Ding2020MMA', 'Sitawarin2020Improving', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial_extra', 'Rebuffi2021Fixing_70_16_cutmix_extra']
# 41.44%; 50.72%; 55.54%; 60.04%; 66.56%
L2_MODEL = ['Ding2020MMA', 'Augustin2020Adversarial', 'Rebuffi2021Fixing_28_10_cutmix_ddpm','Gowal2020Uncovering_extra', 'Rebuffi2021Fixing_70_16_cutmix_extra']
# 66.09%; 72.91%; 78.80%; 80.53%; 82.32%
CORRUPUT_MODEL = ['Addepalli2021Towards_WRN34', 'Kireev2021Effectiveness_RLAT','Kireev2021Effectiveness_RLATAugMixNoJSD','Kireev2021Effectiveness_RLATAugMix', 'Diffenderfer2021Winning_LRR_CARD_Deck']
# 76.78%; 84.10%; 88.53%; 89.60%; 92.78%
def load_cifar10_models(n=1):
    '''
    加载一系列模型来攻击... 消耗很高, 速度很慢
    '''
    models = []
    models.append(load_cifar10_stander_model())
    for i, model_name in enumerate(LINF_MODEL):
        model = load_model(model_name, dataset='cifar10', threat_model='Linf',model_dir=model_path)
        models.append(model)
        if i>= n: break
    for i, model_name in enumerate(L2_MODEL):
        model = load_model(model_name, dataset='cifar10', threat_model='L2',model_dir=model_path)
        models.append(model)
        if i>= n: break
    for i, model_name in enumerate(CORRUPUT_MODEL):
        model = load_model(model_name, dataset='cifar10', threat_model='corruptions',model_dir=model_path)
        models.append(model)
        if i>= n: break
    return models

# models = load_cifar10_models(5)
# print(len(models))


# for downloading data and models~
def download_data(name):
    '''
    作用: 下载数据到root/data下
    '''
    data_path = os.path.join(root, 'data')
    assert name in ['cifar10', 'cifar100', 'imagenet']
    if name == 'cifar10':
        return rb.data.load_cifar10(data_dir=data_path)
    elif name == 'cifar100':
        return rb.data.load_cifar100(data_dir=data_path),
    else:
        return rb.data.load_imagenet(data_dir=data_path)

def download_models():
    '''
    作用: 下载模型到root/model下
    说明: 
    貌似因为下载问题, 每次下载都错误, 所以回报错_pickle.UnpicklingError: invalid load key, '<.'
    需要下载最新版本的robustbenchmark即可, 或者使用官方下载地址(见robustbench的issue)
    '''
    model_path = os.path.join(root, 'model')
    all_models = rb.model_zoo.model_dicts
    for dataset, dataset_models in all_models.items():
        if dataset.value == 'imagenet': # 暂时不下载; 之后记得comment掉
            continue
        for threat, threat_models in dataset_models.items():
            for name in threat_models:
                print(name , dataset.value, threat.value) 
                model = rb.utils.load_model(
                    name, model_dir=model_path,
                    dataset=dataset.value, threat_model=threat.value)
                del model
                
