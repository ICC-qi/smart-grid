import sys
import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm
from .actor import Actor
from .critic import Critic
from utils.stats import gather_stats
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.memory_buffer import MemoryBuffer
import copy
import random

class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, env, dt, act_dim, env_dim, act_range, act_bias, seed, buffer_size = 20000, gamma = 1, lr = 5e-5, tau = 0.001, rewardScale = 2e-3, epsilon = 1, decay = 1):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.act_bias = act_bias
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.rewardScale = rewardScale
        self.epsilon = epsilon
        self.decay = decay
        self.seed = seed
        # Create critic networks
        self.critic = Critic(self.env_dim, act_dim, lr, tau, act_range, act_bias, seed, env)
        self.actor = Actor(self.env_dim, self.act_dim, self.act_range, self.act_bias, 0.1 * self.lr, self.tau, seed, env)
        # save initial weights of actor and critic
        self.actor_init_weight, self.critic_init_weight = self.actor.model.get_weights(), self.critic.model.get_weights()
        self.buffer_size = buffer_size
        self.dt = dt

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_critic(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        # Input: [states, actions], Output: critic_target
        self.critic.train_on_batch(states, actions, critic_target)

 
    def update_actor(self, states, actions):
        """ Update actor and critic networks from sampled experience
        """
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # print("grads::",grads)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))

    def train(self, env, nb_episodes, batch_size, summary_writer = None):
        writer = tf.summary.FileWriter('./logs', tf.get_default_graph())
        writer.close()
        results = []
        days = 14
        t_max = int(24 / self.dt)
        # t_max = 2
        # t = t_max - 1
        tqdm_ts = tqdm(range(t_max), desc='Score', leave=True, unit=" timesteps")
        env.load_data(days)
        state_bias = np.array([env.PL_PPV_bias, env.E_bias])
        state_range = np.array([env.PL_PPV_range, env.E_range])
        for ts in tqdm_ts:
            t = t_max - ts
            # Initialize replay buffer R
            self.buffer = MemoryBuffer(self.buffer_size)
            noise = OrnsteinUhlenbeckProcess(sigma=0.5, n_steps_annealing=nb_episodes, size=self.act_dim, seed=self.seed)
            summary_writer = tf.summary.FileWriter(
                "./tensorboard_TS/tensorboard_" + "TS_" + str(int(t)))
            # First, gather experience
            eps = range(nb_episodes)
            history_norm = env.getHistory(t)
            history_norm = np.reshape(history_norm, [env.hisStep, env.stateDim])
            for e in eps:
                # cumul_reward, done = 0, False
                day = np.random.randint(0, days)
                old_state = env.reset(t - 1 + int(24 / self.dt) * day)

                battery = tfSummary('battery', old_state[1])
                summary_writer.add_summary(battery, global_step=e)

                # if e==0:
                #     old_state = env.reset(t - 1)
                # else:
                #     E_t = random.randint(env.E_min, env.E_max)
                #     old_state[2] = E_t
                #     env.E_t = E_t
                #     env.state = old_state

                # Select action a_t according to the current policy and exploration noise

                old_state_norm = (old_state - state_bias) / state_range
                history_norm[-1, -1] = old_state_norm[1]
                history1_norm = history_norm.copy()
                # old_state_norm = np.expand_dims(history_norm, axis=0)

                a_norm = self.policy_action(history1_norm)

                a = a_norm * self.act_range + self.act_bias


                ou_noise = self.epsilon * noise.generate(e)
                noise_summary = tfSummary('noise', ou_noise)
                summary_writer.add_summary(noise_summary , global_step=e)


                # print ("ou_noise:", ou_noise)
                a = np.clip(a + ou_noise, self.act_bias - self.act_range, self.act_bias + self.act_range)
                # Retrieve new state, reward, and whether the state is terminal
                # new_state, r, done, _ = env.step(a)

                # if e==0:
                #     new_state, r, done, _ = env.step(a)
                # else:
                #     E_t, r = env.update_e_and_r(a)
                #     env.E_t = E_t
                #     new_state[2] = E_t

                new_state, r, done, _ = env.step(a)
                new_state_norm = (new_state - state_bias) / state_range

                r *= self.rewardScale
                # Add outputs to memory buffer
                # if e!=0:
                #     states, actions, rewards, dones, new_states, __ = self.sample_batch(batch_size - 1)
                # else:
                #     states, actions, rewards, dones, new_states, __ = np.zeros([2,1]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
                # states = np.append(states, old_state)
                # actions = np.append(actions, a)
                # rewards = np.append(rewards, r)
                # dones = np.append(dones, done)
                # new_states = np.append(new_states, new_state)
                # self.memorize(old_state, a, r, done, new_state)

                if e==0: # e=0, save to buffer at first
                    self.memorize(history1_norm, a_norm, r, done, new_state_norm)
                    histories_norm, actions_norm, rewards, dones, new_states_norm, __ = self.sample_batch(batch_size - 1)
                else:
                    histories_norm, actions_norm, rewards, dones, new_states_norm, __ = self.sample_batch(batch_size - 1)
                    # states = np.r_[states, old_state.reshape([1, 3])]
                    histories_norm = np.r_[histories_norm, history_norm.reshape([1, env.hisStep, env.stateDim])]
                    actions_norm = np.r_[actions_norm, a_norm.reshape([1,1])]
                    rewards = np.r_[rewards, r.reshape([1,1])]
                    dones = np.r_[dones, done]
                    # new_states = np.r_[new_states, new_state.reshape([1,3])]
                    new_states_norm = np.r_[new_states_norm, new_state_norm.reshape([1,2])]
                    self.memorize(history1_norm, a_norm, r, done, new_state_norm)

                new_states = new_states_norm * state_range + state_bias

                if t == t_max:
                    # Select action according to the myopic policy
                    # r_Ts = []
                    # for new_state_T_1 in new_states:
                    #     r_T, a_T = env.myopic_policy(new_state_T_1)
                    #     r_Ts.append(r_T)
                    # r_Ts = np.expand_dims(r_Ts, axis=1)
                    # critic_target = rewards + self.gamma * r_Ts
                    critic_target = rewards
                else:
                    # Predict target q-values using target networks
                    histories1_norm = histories_norm.copy()
                    histories_t1_norm = np.reshape(histories1_norm,[np.shape(histories1_norm)[0], env.hisStep * env.stateDim])
                    histories_t1_norm[:, 0: -env.stateDim - 1] = histories_t1_norm[:, env.stateDim:-1]
                    histories_t1_norm[:, -env.stateDim:] = new_states_norm[:, :]
                    histories_t1_norm = np.reshape(histories_t1_norm, [np.shape(histories_t1_norm)[0], env.hisStep, env.stateDim])

                    actions_t1_norm = self.actor.target_predict(histories_t1_norm)
                    q_values = self.critic.target_predict([histories_t1_norm, actions_t1_norm])
                    critic_target = self.bellman(rewards, q_values, dones)

                    # Export results for Tensorboard
                    q = tfSummary('q-value', q_values[0])
                    summary_writer.add_summary(q, global_step=e)
                    summary_writer.flush()

                # Train both networks on sampled batch, update target networks
                self.update_critic(histories_norm, actions_norm, critic_target)
                self.update_actor(histories_norm, actions_norm)
                if e > 100:
                    self.epsilon *= self.decay
                # cumul_reward += r
                cumul_reward = critic_target[-1]

                # Export results for Tensorboard
                score = tfSummary('score', cumul_reward)
                summary_writer.add_summary(score, global_step=e)

                # print("Score: " + str(cumul_reward))

            # Display score
            tqdm_ts.set_description("Score: " + str(cumul_reward))
            tqdm_ts.refresh()
            # update the target network
            self.critic.transfer_weights()
            self.actor.transfer_weights()
            # save weight of actor network
            exp_dir = './DDPG/models/'
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            export_path = '{}{}_TS_{}'.format(exp_dir,"DDPG",str(t))
            self.save_actor_weights(export_path)
            self.save_critic_weights(export_path)
            # reset the weight of actor and critic
            self.actor.model.set_weights(self.actor_init_weight)
            self.critic.model.set_weights(self.critic_init_weight)
            # t = t - 1

        return results

    # def save_weights(self, path):
    #     path += '_LR_{}'.format(self.lr)
    #     self.actor.save(path)
    #     self.critic.save(path)

    def save_actor_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)

    def save_critic_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.critic.save(path)

    # def load_weights(self, path_actor, path_critic):
    #     self.critic.load_weights(path_critic)
    #     self.actor.load_weights(path_actor)
