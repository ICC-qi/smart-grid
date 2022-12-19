""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import tensorflow as tf
from ENV import env_mg


from DDPG.ddpg import DDPG

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

# from utils.atari_environment import AtariEnvironment
# from utils.continuous_environments import Environment
from utils.networks import get_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def main(args=None):
    env_name="Microgrid-v0"
    gpu=0
    # nb_episodes = 30000
    # batch_size = 128
    nb_episodes = 6
    batch_size = 3

    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = 1
    set_session(get_session())
    # summary_writer = tf.summary.FileWriter("./tensorboard_ENV_" +env_name + "_EP_" + str(nb_episodes) + "_BS_" + str(batch_size))

    # Environment Initialization
    env = env_mg()
    dt = env.dt
    state_dim = env.stateDim
    action_dim = env.actionDim
    maxAction = env.action_max
    minAction = env.action_min
    act_bias = (maxAction + minAction) / 2.0
    act_range = maxAction - act_bias

    algo = DDPG(env, dt, action_dim, state_dim, act_range, act_bias, env.seed)


    # Train
    stats = algo.train(env, nb_episodes, batch_size)

    # Export results to CSV
    # if(args.gather_stats):
    #     df = pd.DataFrame(np.array(stats))
    #     df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # # Save weights
    # exp_dir = '{}/models/'.format("DDPG")
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)


def baseline_random():
    env = env_mg()
    env.baseline_random_run(1)

def baseline_local_opt():
    env = env_mg()
    env.baseline_local_opt_run(100)

def baseline_maxpower_run():
    env = env_mg()
    env.baseline_maxpower_run(1)

if __name__ == "__main__":
    # main()
    # baseline_random()
    baseline_local_opt()
    # baseline_maxpower_run()
