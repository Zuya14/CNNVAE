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

def collect_init_episode(stateBuffer, memory_size, collect_num, min_step, id=0, clientReset=False):
    env = gym.make('vaernn-v0')
    env.setting()
    memory = EpisodeMemory(mem_size=memory_size)

    collect_count = 0

    while collect_count < collect_num:
        episode = Episode()

        state = stateBuffer.sample(delete_flag=False)
        x, y, theta, vx, vy, w = state.values()
        env.reset(x, y, theta, vx, vy, w, clientReset)

        step = 0
        while True:
            action = env.sample_random_action()
            observation, reward, done, _ = env.step(action)
            
            episode.append(observation, action, reward, done)
            step += 1
            if done:
                break
        
        if step >= min_step:
            collect_count += 1
            memory.append(episode)

        # print(state)
        # print(id, ":", step, ", ", collect_count)

    # print("end", id)

    return memory

def collect_episode(stateBuffer, memory_size, state_buffer_size, collect_num, min_step, id=0, clientReset=False, sample_rate=0.001, sec=0.01):
    env = gym.make('vaernn-v0')
    env.setting(sec)
    memory = EpisodeMemory(mem_size=memory_size)

    new_stateBuffer = StateBuffer(max_size=state_buffer_size)

    collect_count = 0

    while collect_count < collect_num:
        episode = Episode()

        state = stateBuffer.sample(delete_flag=False)
        x, y, theta, vx, vy, w, action = state.values()
        env.reset(x, y, theta, vx, vy, w, action, clientReset)
        observation = env.observe()

        step = 0
        while True:
            pre_action = env.sample_random_action()
            next_observation, reward, done, _ = env.step(pre_action)

            action = env.sim.action

            if random.random() < sample_rate:
                x, y, theta, vx, vy, w = env.getState()
                stateMem = StateMem(x, y, theta, vx, vy, w, action)
                new_stateBuffer.append(stateMem)
            
            episode.append(observation, action, reward, done)
            observation = next_observation
            
            step += 1
            if done:
                break
        
        if step >= min_step:
            collect_count += 1
            memory.append(episode)

        # print(state)
        # print(id, ":", step, ", ", collect_count)

    # print("end", id)

    return memory, new_stateBuffer

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
        # while True:
        for i in range(400):
            # pre_action = env.sample_random_action()
            next_observation, reward, done, _ = env.step(pre_action)

            action = env.sim.action

            # if random.random() < sample_rate:
            #     x, y, theta, vx, vy, w = env.getState()
            #     stateMem = StateMem(x, y, theta, vx, vy, w, action)
            #     new_stateBuffer.append(stateMem)
            
            episode.append(observation[:1080], action, reward, done)
            observation = next_observation
            
            step += 1
            if done:
                break
        
        # if step >= min_step:
        #     collect_count += 1
        #     memory.append(episode)

        memory.append(episode)

        # print(state)
        # print(id, ":", step, ", ", collect_count)

    # print("end", id)

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

    parser.add_argument('--threads', type=int, default=os.cpu_count())

    parser.add_argument("--sec", type=float, default=0.01)

    args = parser.parse_args()

    out_dir = './result-VGGVAE_single' 

    if args.id != '':
        out_dir += '/' + args.id

    out_dir += '/channels'

    for channel in args.channels:
        out_dir += '_{}'.format(channel)
    out_dir += '_latent_{}'.format(args.latent)

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

    # with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     # future_list = [executor.submit(collect_init_episode, memory_size=args.memory_size, stateBuffer=stateBuffer, collect_num=args.init_episode, min_step=args.min_step, id=i, clientReset=True) for i in range(args.threads)]
    #     # future_list = [executor.submit(collect_episode,stateBuffer=stateBuffer,  memory_size=args.memory_size, state_buffer_size=args.state_buffer_size, collect_num=args.init_episode, min_step=args.min_step, id=i, clientReset=True) for i in range(args.threads)]
    #     future_list = [executor.submit(collect_episode, stateBuffer=stateBuffer,  memory_size=args.memory_size, state_buffer_size=args.state_buffer_size, collect_num=args.init_episode, min_step=args.min_step, id=i, clientReset=True, sample_rate=sample_rate, sec=args.sec) for i in range(args.threads)]
    #     future_return = futures.as_completed(fs=future_list)

    # for future in future_return:
    #     mem, state_buf = future.result()
    #     # print(mem.size, state_buf.size())

    #     memory.extend(mem)
    #     stateBuffer.extend(state_buf)

    for i in range(args.threads):
        # mem, state_buf = collect_episode(stateBuffer=stateBuffer,  memory_size=args.memory_size, state_buffer_size=args.state_buffer_size, collect_num=args.init_episode, min_step=args.min_step, id=i, clientReset=True, sample_rate=sample_rate, sec=args.sec)
        mem, state_buf = collect_episode_over(stateBuffer=stateBuffer,  memory_size=args.memory_size, state_buffer_size=args.state_buffer_size, collect_num=args.init_episode, min_step=args.min_step, id=i, clientReset=True, sample_rate=sample_rate, sec=args.sec)
        memory.extend(mem)
        stateBuffer.extend(state_buf)

    ''' ---- Train ---- '''

    print("Train", datetime.datetime.now())

    num_epochs = args.epochs

    vae_train =  VGGVAE_trainer(args.channels, args.latent, args.cnn_outsize, device=device)

    if args.models is not '' and os.path.exists(args.models):
        vae_train.load(args.models)

    train_plot_data = plot_graph.Plot_Graph_Data(out_dir, 'train_loss', {'train_loss': [], 'mse_loss': [], 'KLD_loss': []})
    test_plot_data  = plot_graph.Plot_Graph_Data(out_dir, 'test_loss',  {'test_loss': []})
    plotGraph = plot_graph.Plot_Graph([train_plot_data, test_plot_data])

    for epoch in range(1, num_epochs+1):

        # with futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        #     future_list = [executor.submit(collect_episode, stateBuffer=stateBuffer,  memory_size=args.memory_size, state_buffer_size=args.state_buffer_size, collect_num=1, min_step=args.min_step, id=i, clientReset=True, sample_rate=sample_rate, sec=args.sec) for i in range(args.threads)]

        # datas_observations = [memory.sample2_obs(n=args.batch_size, L=args.chunk_size) for _ in range(args.collect_interval)]
        datas_observations = [memory.sample2_obs(n=args.batch_size, L=1) for _ in range(args.collect_interval)]

        mse_loss, KLD_loss = vae_train.train(datas_observations)
        loss = mse_loss + KLD_loss

        datas_observations = [memory.sample2_obs(n=args.test_batch_size, L=args.chunk_size)]
        test_loss = vae_train.test(datas_observations)

        plotGraph.addDatas('train_loss', ['train_loss', 'mse_loss', 'KLD_loss'], [loss, mse_loss, KLD_loss])
        plotGraph.addDatas('test_loss', ['test_loss'], [test_loss])

        if epoch%10 == 0:
            vae_train.save(out_dir+'/vae.pth')

            plotGraph.plot('train_loss')
            plotGraph.plot('test_loss')

            print('epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}'.format(
                epoch + 1,
                num_epochs,
                loss,
                test_loss))       

        if epoch % (num_epochs//10) == 0:
            vae_train.save(out_dir+'/vae{}.pth'.format(epoch))
            
            data = memory.sample2(n=args.test_batch_size, L=args.chunk_size)[0].to(device)
            data = data.to(device).view(-1, 1, 1080)
            recon_x, mu, logvar = vae_train.vae(data)
            save_image(torch.cat([data.view(-1,1080), recon_x.view(-1,1080)], dim=1), '{}/result{}.png'.format(out_dir, epoch))
            
            print("save:epoch", epoch)

        #     future_return = futures.as_completed(fs=future_list)

        # for future in future_return:
        #     mem, state_buf = future.result()
        #     # print(mem.size, state_buf.size())

        #     memory.extend(mem)
        #     stateBuffer.extend(state_buf)

    e_time = datetime.datetime.now()

    print("start:", s_time)
    print("end:", e_time)
    print("end-start:", e_time-s_time)







