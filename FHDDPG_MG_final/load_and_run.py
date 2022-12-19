""" Load and display pre-trained model in OpenAI Gym Environment
"""

import os
import sys
# import gym
import argparse
import numpy as np
# import pandas as pd
import tensorflow as tf
from ENV import env_mg


from DDPG.ddpg import DDPG

from keras.backend.tensorflow_backend import set_session
# from keras.utils import to_categorical
#
# from utils.atari_environment import AtariEnvironment
# from utils.continuous_environments import Environment
from utils.networks import get_session
import matplotlib.pyplot as plt
from scipy.stats import kde
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess

# gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDPG',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    #
    parser.add_argument('--model_path', type=str, help="Number of training episodes")
    parser.add_argument('--actor_path', type=str, help="Number of training episodes")
    parser.add_argument('--critic_path', type=str, help="Batch size (experience replay)")
    #
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def load_and_print(args=None):

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(get_session())

    num_episode = 100

    env = env_mg()
    dt = env.dt
    state_dim = env.stateDim
    action_dim = env.actionDim
    maxAction = env.dg_max
    minAction = env.dg_min
    act_bias = (maxAction + minAction) / 2.0
    act_range = maxAction - act_bias


    algo = DDPG(dt, action_dim, state_dim, act_range, act_bias, env.seed)

    # Display agent
    # pv = []
    # pl = []
    dpvl = []
    e = []
    dg = []
    cus = []
    cdg = []
    totalReward_Avg = 0

    ad = 0.005
    bd = 6
    cd = 100
    train_days = 21
    load_days = 1
    env.load_data(load_days)
    env.get_range_bias(train_days)
    q_true = []
    q_estimated = []
    for i in range(num_episode):
        day = np.random.randint(0, load_days)
        r_true = []
        totalReward = 0
        state, time = env.reset(int(24 / dt) * day), 0
        state_bias = np.array([env.PL_PPV_bias, env.E_bias])
        state_range = np.array([env.PL_PPV_range, env.E_range])
        state_norm = (state - state_bias) / state_range
        t_max = int(24 / dt)
        # t_max = 2
        while time < t_max:
            # pl.append(state[0])
            # print("Load:", state[0])
            # pv.append(state[1])
            # print("PV:", state[1])
            # e.append(state[2])
            # print("E:", state[2])

            dpvl.append(state_norm[0])
            print("Load-PV:", state_norm[0])
            e.append(state_norm[1])
            print("E:", state_norm[1])

            if time < t_max - 1:
                # path_actor = '../models_and_tb/FHDDPG-14days/seed' + str(env.seed) + '/models/DDPG' + '_TS_' + str(time + 1) + '_LR_' + str(algo.lr) + '_actor.h5'
                # path_critic = '../models_and_tb/FHDDPG-14days/seed' + str(env.seed) + '/models/DDPG' + '_TS_' + str(time + 1) + '_LR_' + str(algo.lr) + '_critic.h5'
                # path_actor = './DDPG/models/DDPG' + '_TS_' + str(time + 1) + '_LR_' + str(algo.lr) + '_actor.h5'
                # path_critic = './DDPG/models/DDPG' + '_TS_' + str(time + 1) + '_LR_' + str(algo.lr) + '_critic.h5'
                model_seed = 1
                path_actor = '../models_and_tb/FHDDPG-21days/seed' + str(model_seed) + '/models/DDPG' + '_TS_' + str(time + 1) + '_LR_' + str(algo.lr) + '_actor.h5'
                path_critic = '../models_and_tb/FHDDPG-21days/seed' + str(model_seed) + '/models/DDPG' + '_TS_' + str(time + 1) + '_LR_' + str(algo.lr) + '_critic.h5'
                algo.actor.load_weights(path_actor)
                algo.critic.load_weights(path_critic)
                a_norm = algo.policy_action(state_norm)

                a = a_norm * act_range + act_bias

                state, r, done, _ = env.step(a)
                state_norm = (state - state_bias) / state_range
#                env.cnt += 1
                r *= env.rewardScale
                totalReward += r

                dg.append(a)
                print("DG:", a)
                cus.append(env.cus[time])
                print("CUS:", env.cus[time])
                cdg.append(env.cdg[time])

                # evaluate critic
                r_true.append(r[0])
                state_norm_for_q = state_norm.copy()
                state_norm_for_q = np.expand_dims(state_norm_for_q, axis=0)
                q_values = algo.critic.target_predict([state_norm_for_q, algo.actor.target_predict(state_norm_for_q)])
                q_estimated.append(q_values[0][0])
            else:
                r, a = env.myopic_policy(state)
                totalReward += r
                r_true.append(r)

                dg.append(a)
                print("DG:", a)
                cus.append(env.cus[time])
                print("CUS:", env.cus[time])
                cdg.append(env.cdg[time])


            time += 1

        print("Episode:", i, ",score:", totalReward)
        totalReward_Avg = totalReward_Avg + totalReward

        # compute q_true
        for i in range(len(r_true)-1):
            q_true.append(sum(r_true[i+1:]))

    totalReward_Avg = totalReward_Avg / num_episode
    print("average score:", totalReward_Avg)

    # plot q_true vs q_estimated
    print("q_true:", q_true)
    print("q_estimated:", q_estimated)

    # Calculate the point density
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    q_true = np.array(q_true)
    q_estimated = np.array(q_estimated)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 15,
            }

    identity_line = np.linspace(-0.4, 0)
    plt.plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=1.0)
    plt.xlim([-0.3, 0])
    plt.ylim([-0.3, 0])
    plt.xlabel("Return", font)
    plt.ylabel("Estimated Q", font)
    plt.grid()

    nbins = 20
    xi, yi = np.mgrid[q_true.min():q_estimated.max():nbins * 1j, q_estimated.min():q_estimated.max():nbins * 1j]
    k = kde.gaussian_kde(np.array([q_true, q_estimated]))
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Greys)
    plt.contour(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greys)

    plt.show()
    fig.savefig('./fig_q.png', dpi=300)

    

    # x = range(0, int(24/dt))
    # x = range(0, 1)
    # plt.subplot(611)
    # plt.plot(x, pv, marker='.', mec='r', mfc='w')
    # plt.xlabel("step")
    # plt.ylabel("PV")

    # plt.subplot(612)
    # plt.plot(x, pl, marker='.', mec='r', mfc='w')
    # plt.xlabel("step")
    # plt.ylabel("Load")

    # plt.subplot(613)
    # plt.plot(x, e, marker='.', mec='r', mfc='w')
    # plt.xlabel("step")
    # plt.ylabel("battery SOC")

    # plt.subplot(614)
    # plt.plot(x, dg, marker='.', mec='r', mfc='w')
    # plt.xlabel("step")
    # plt.ylabel("DG power")

    # plt.subplot(615)
    # plt.plot(x, cus, marker='.', mec='r', mfc='w')
    # plt.xlabel("step")
    # plt.ylabel("CUS_t")

    # plt.subplot(616)
    # plt.plot(x, cdg, marker='.', mec='r', mfc='w')
    # plt.xlabel("step")
    # plt.ylabel("CDG_t")

    # plt.show()

def load_and_train(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    env_name = "Microgrid-v0"
    gpu = 0
    nb_episodes = 1500
    batch_size = 64

    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = 1
    set_session(get_session())
    summary_writer = tf.summary.FileWriter("./tensorboard_ENV_" +env_name + "_EP_" + str(nb_episodes) + "_BS_" + str(batch_size))

    env = env_mg()
    dt = env.dt
    state_dim = env.stateDim
    action_dim = env.actionDim
    maxAction = env.dg_max
    minAction = env.dg_min
    act_bias = (maxAction + minAction) / 2.0
    act_range = maxAction - act_bias

    algo = DDPG(dt, action_dim, state_dim, act_range, act_bias, env.seed)
    algo.load_weights(args.actor_path, args.critic_path)

    stats = algo.train(env, nb_episodes, batch_size, summary_writer)

    exp_dir = '{}/models/'.format("DDPG")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_ENV_{}'.format(exp_dir, "DDPG", "MG")

    algo.save_weights(export_path)


if __name__ == "__main__":
    load_and_print()
    # load_and_train()
