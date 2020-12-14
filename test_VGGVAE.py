import argparse

import math
import numpy as np
import random
import os
import datetime
from concurrent import futures

from StateBuffer import StateMem, StateBuffer
from EpisodeMemory import Episode, EpisodeMemory
from lidar_util import imshowLocalDistance

import pybullet as p
import gym
import cv2

import torch
from torchvision.utils import save_image

from VGGVAE_trainer import VGGVAE_trainer
import plot_graph

from gym.envs.registration import register

register(
    id='vaernn-v0',
    entry_point='vaernnEnv0:vaernnEnv0'
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collect_episode_over(stateBuffer, memory_size, state_buffer_size, collect_num, min_step, id=0, clientReset=False, sample_rate=0.001, sec=0.01):
    env = gym.make('vaernn-v0')
    env.setting(sec)
    memory = EpisodeMemory(mem_size=memory_size)

    new_stateBuffer = StateBuffer(max_size=state_buffer_size)

    collect_count = 0

    # while collect_count < collect_num:
    while stateBuffer.size() > 0:

        episode = Episode()

        # state = stateBuffer.sample(delete_flag=False)
        state = stateBuffer.sample(delete_flag=True)
        x, y, theta, vx, vy, w, action = state.values()
        env.reset(x, y, theta, vx, vy, w, action, clientReset)
        observation = env.observe()

        print(action)
        pre_action = np.array([0.5, 0.5, 0.5])

        step = 0
        for i in range(400):
            next_observation, reward, done, _ = env.step(pre_action)

            action = env.sim.action
            
            episode.append(observation[:1080], action, reward, done)
            observation = next_observation
            
            step += 1
            if done:
                break

        memory.append(episode)

    return memory, new_stateBuffer

if __name__ == '__main__':

    s_time = datetime.datetime.now()

    print("start:", s_time)

    parser = argparse.ArgumentParser(description='train')

    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--state-buffer-size", type=int, default=1000)
    parser.add_argument("--init-episode", type=int, default=1)
    parser.add_argument("--min_step", type=int, default=10)

    parser.add_argument("--channels", nargs="*", type=int, default=[1, 64])
    parser.add_argument("--cnn-outsize", type=int)
    parser.add_argument("--latent", type=int, default=18)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--batchnorm', action='store_true')
    # parser.add_argument('--adabelief', action='store_true')

    parser.add_argument('--chunk-size', type=int, default=50)
    parser.add_argument('--collect-interval', type=int, default=10)

    parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
    parser.add_argument('--id', type=str, default='')

    parser.add_argument('--threads', type=int, default=1)

    parser.add_argument("--sec", type=float, default=0.01)

    args = parser.parse_args()


    out_dir = './result-VGGVAE' 

    if args.id != '':
        out_dir += '/' + args.id

    out_dir += '/channels'

    for channel in args.channels:
        out_dir += '_{}'.format(channel)
    out_dir += '_latent_{}'.format(args.latent)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ''' ---- Initialize ---- '''

    # env0 = gym.make('vaernn-v0')
    # env0.setting()
    
    memory = EpisodeMemory(mem_size=args.memory_size)

    p4 = math.pi / 4.0
    # xs = [-3.0, -1.5, 0.0, 1.5, 3.0]
    xs = [0.0]
    # ys = [-3.0, 0.0, 3.0]
    ys = [0.0]
    # thetas = [0.0, p4, 2.0*p4, 3.0*p4, 4.0*p4, 5.0*p4, 6.0*p4, 7.0*p4]
    thetas = [0.0]

    stateBuffer = StateBuffer(max_size=args.state_buffer_size)
    # stateBuffer.generate(xs, ys, thetas)
    stateBuffer.generate2(xs, ys, thetas)

    ''' ---- Initialize EpisodeMemory ---- '''

    print("Initialize EpisodeMemory", datetime.datetime.now())

    sample_rate = args.sec if args.sec < 1 else 1 

    for i in range(args.threads):
        mem, state_buf = collect_episode_over(stateBuffer=stateBuffer,  memory_size=args.memory_size, state_buffer_size=args.state_buffer_size, collect_num=args.init_episode, min_step=args.min_step, id=i, clientReset=True, sample_rate=sample_rate, sec=args.sec)

        memory.extend(mem)
        stateBuffer.extend(state_buf)

    ''' ---- Train ---- '''

    print("Test", datetime.datetime.now())

    num_epochs = args.epochs

    vae_train =  VGGVAE_trainer(args.channels, args.latent, args.cnn_outsize, device=device)

    if args.models != '' and os.path.exists(args.models):
        print("load:", args.models)
        vae_train.load(args.models)
    else:
        exit()

    for index in range(len(memory.episodes)):
        datas_observations = torch.tensor(memory.get(index)[0]).float()
        print(datas_observations.size())
        
        data = datas_observations.to(device).view(-1, 1, 1080)
        recon_x, mu, logvar = vae_train.vae(data)
        save_image(torch.cat([data.view(-1,1080), recon_x.view(-1,1080)], dim=1), '{}/test_recon{}.png'.format(out_dir, index))

        recon_x = recon_x.view(-1,1080).cpu().detach().numpy()
        np.savetxt('{}/recon{}.csv'.format(out_dir, index), recon_x, delimiter=',')

    e_time = datetime.datetime.now()

    print("start:", s_time)
    print("end:", e_time)
    print("end-start:", e_time-s_time)
