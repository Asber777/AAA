import os, torch, argparse, shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from sub_attacker import SubAttacker
# from torch.distributions import Categorical
from resnet import ResNet18
from data_model import load_cifar10_data, load_cifar10_stander_model

def get_args():
    parser = argparse.ArgumentParser('AAA')
    parser.add_argument('--exp_dir', type=str, default='exp',
                        help='which folder to store experiments')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
    parser.add_argument('--num_steps', type=int, default=5, metavar='N',
                    help='max episode length (default: 5)')
    parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 2000)')
    parser.add_argument('--ckpt_freq', type=int, default=100, 
		            help='model saving frequency')
    parser.add_argument('--agent_lr', type=float, default=0.01, metavar='LR',
                    help='learning rate of reinforce agent')
    parser.add_argument('--nb_iter', type=int, default=2, 
		            help='how much iteration in each subattacker')
    # parser.add_argument('--ntrain', type=int, default=30000, 
    #                 help='length of train dataset')
    # parser.add_argument('--nval', type=int, default=10000, 
    #                 help='length of val dataset')
    args = parser.parse_args()
    return args

def create_experiment_folder(args):
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    args.savedir = "{}".format(args.num_steps)
    args.savedir = os.path.join(args.exp_dir, args.savedir)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    shutil.copy('AAA_abs.py', args.savedir)

args = get_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
create_experiment_folder(args)

def select_action(policy_model, state):
    probs = F.softmax(policy_model(state), dim=1)
    action = probs.multinomial(num_samples=1).data
    prob = probs[:, action[0,0]].view(1, -1)
    log_prob = prob.log()
    entropy = - (probs*probs.log()).sum()
    return action[0], log_prob, entropy

def Get_reward_done(input_batch, y_batch, target_model):
    if target_model(input_batch).argmax(dim=1) == y_batch:
        return -1, False
    else:
        return +5, True

def step(action, x, y, net, attacker:SubAttacker):
    next_state = attacker.attack(action, x, y, net)
    reward, done = Get_reward_done(next_state, y, net)
    return next_state, reward, done, {}

def update_parameters(rewards, log_probs, entropies, gamma):
    R = torch.zeros(1, 1)
    loss = 0
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
    loss = loss / len(rewards)
    return loss


def main():
    attacker = SubAttacker(args.nb_iter)
    agent = ResNet18(attacker.num_op, input_channel=3).cuda()
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=args.agent_lr)
    train_loader = load_cifar10_data(train=False, batch_size=1)
    model = load_cifar10_stander_model().cuda().eval()
    for i_episode in range(args.num_episodes):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            state, targets = inputs.cuda(), targets.cuda()
            entropies = []
            log_probs = []
            rewards = []
            for t in range(args.num_steps):
                action, log_prob, entropy = select_action(agent, state)
                action = action.cpu().detach().clone().item()
                next_state, reward, done, _ = step(action, state, targets, model, attacker)
                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                if done: break
            reinforce_loss = update_parameters(rewards, log_probs, entropies, args.gamma)
            optimizer.zero_grad()
            reinforce_loss.backward()
            utils.clip_grad_norm(agent.parameters(), 40)
            optimizer.step()
            
        if i_episode % args.ckpt_freq == 0:
            torch.save(agent.state_dict(), os.path.join(args.savedir, 'reinforce-'+str(i_episode)+'.pkl'))

        print("Episode: {}, reward: {}".format(batch_idx, np.sum(rewards)))

main()