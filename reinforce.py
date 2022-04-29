# encoding: utf-8
import argparse
import numpy as np
import copy
import os
import numpy
import time
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

logger = logging.getLogger(__name__)
CUDA_LAUNCH_BLOCKING=1 # for debugging when GPU paralleling. 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import json
import shutil
def get_args():
    parser = argparse.ArgumentParser('AAA')
    parser.add_argument('--exp_dir', type=str, default='exp',
                        help='which folder to store experiments')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                        help='discount factor for reward (default: 1.)')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_steps', type=int, default=5, metavar='N',
                        help='max episode length (default: 5)')
    parser.add_argument('--num_episodes', type=int, default=5000, metavar='N',
                        help='number of episodes (default: 2000)')
    parser.add_argument('--ckpt_freq', type=int, default=100, 
                        help='model saving frequency')
    parser.add_argument('--clip_grad_norm', default=1.0, type=float, 
                        help='set norm limit on gradient of policy')
    parser.add_argument('--bs', type=int, default=100, 
                        help='attack batch-size')
    parser.add_argument('--ntrain', type=int, default=30000, 
                        help='length of train dataset')
    parser.add_argument('--nval', type=int, default=10000, 
                        help='length of val dataset')
    args = parser.parse_args()

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # change name if your need to emphasize somthing.
    args.savedir = "{}".format(args.num_steps)
    subfolder = os.path.join(args.exp_dir, args.savedir)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # save json in case parameter set on terminal
    info_json = json.dumps(vars(args), sort_keys=False, indent=4, separators=(' ', ':'))
    with open('{}/{}-info.json'.format(args.savedir, args.time), 'w') as f:
        f.write(info_json)

    logfile = os.path.join(args.savedir, 'output.log')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    shutil.copy('reinforce.py', args.savedir)
    return args

from resnet import ResNet18

from torchvision import datasets as ds
from torchvision import transforms as tfs
from torch.utils.data import DataLoader, Subset

def get_loader(bs, ntrain, nval):
    data_loc = '/home/.faa/data'
    trainset = ds.CIFAR10(root=data_loc, train=True, download=True, transform=tfs.ToTensor())
    testset = ds.CIFAR10(root=data_loc, train=False, download=True, transform=tfs.ToTensor())
    trainset_ = Subset(trainset, np.arange(0, ntrain))        
    valset = Subset(trainset, np.arange(ntrain, ntrain+nval))
    train = DataLoader(trainset_, bs, shuffle=True, pin_memory=True, drop_last = True)
    val = DataLoader(valset, bs, shuffle=True, pin_memory=True, drop_last = True) 
    test = DataLoader(testset, bs, shuffle=True, pin_memory=True, drop_last = True)
    return train, val, test

def get_action_space():
    pass

class REINFORCE:
    def __init__(self, input_channel, action_space):
        self.action_space = action_space
        self.model = ResNet18(len(action_space), input_channel)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.model.train()

    def select_action(self, x):
        probs = F.softmax(self.model(x), dim=1)
        action = probs.multinomial(n_samples=1).data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action[0], log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

def env_step(action):
    pass

if __name__ == '__main__':
    args = get_args()
    train_loader, val_loader, test_loader = get_loader(args.bs, args.ntrain, args.nval)
    action_space = get_action_space()
    agent = REINFORCE(input_channel=3, action_space=action_space)
    for i_episode in range(args.num_episodes):
        # reset
        for x, y in train_loader:
            state, y = x.to(device), y.to(device)
            break
        # entropies_bs = [[] for i in range(args.bs)]
        # log_probs_bs = [[] for i in range(args.bs)]
        # rewards_bs = [[] for i in range(args.bs)]
        entropies = []
        log_probs = []
        rewards = []
        for t in range(args.num_steps):
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, _ = env_step(action.cpu().numpy())
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            if done: break
        agent.update_parameters(rewards, log_probs, entropies, args.gamma)
        if i_episode%args.ckpt_freq == 0:
            torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-'+str(i_episode)+'.pkl'))
        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
    # env.close()

