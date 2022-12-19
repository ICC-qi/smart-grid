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

    def __init__(self, dt, act_dim, env_dim, act_range, act_bias, seed, buffer_size = 20000, gamma = 1, lr = 5e-5, tau = 0.001, rewardScale = 2e-3, epsilon = 1, decay = 1):
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
        self.critic = Critic(self.env_dim, act_dim, lr, tau, act_range, act_bias, seed)
        self.actor = Actor(self.env_dim, self.act_dim, self.act_range, self.act_bias, 0.1 * self.lr, self.tau, seed)
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
        days = 7
        t_max = int(24 / self.dt)
        # t_max = 2
        # t = t_max - 1
        tqdm_ts = tqdm(range(t_max - 1), desc='Score', leave=True, unit=" timesteps")
        env.load_data(days)
        state_bias = np.array([env.PL_PPV_bias, env.E_bias])
        state_range = np.array([env.PL_PPV_range, env.E_range])
        for ts in tqdm_ts:
            t = t_max - ts - 1
            # Initialize replay buffer R
            self.buffer = MemoryBuffer(self.buffer_size)
            noise = OrnsteinUhlenbeckProcess(sigma=0.5, n_steps_annealing=nb_episodes, size=self.act_dim, seed=self.seed)
            summary_writer = tf.summary.FileWriter(
                "./tensorboard_TS/tensorboard_" + "TS_" + str(int(t)))
            # First, gather experience
            eps = range(nb_episodes)
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
                a_norm = self.policy_action(old_state_norm)

                a = a_norm * self.act_range + self.act_bias


                ou_noise = self.epsilon * noise.generate(e)  # 这里原本是ou_noise=self.epsilon * noise.generate(time)
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

                # if e==0: # e=0 save at first
                #     self.memorize(old_state, a, r, done, new_state)
                #     states, actions, rewards, dones, new_states, __ = self.sample_batch(batch_size - 1)
                # else:
                #     states, actions, rewards, dones, new_states, __ = self.sample_batch(batch_size - 1)
                #     np.append(states, old_state, axis=None)
                #     np.append(actions, a, axis=None)
                #     np.append(rewards, r, axis=None)
                #     np.append(dones, done, axis=None)
                #     np.append(new_states, new_state, axis=None)
                #     self.memorize(old_state, a, r, done, new_state)

                if e==0: # e=0 save to the buffer at first
                    self.memorize(old_state_norm, a_norm, r, done, new_state_norm)
                    states_norm, actions_norm, rewards, dones, new_states_norm, __ = self.sample_batch(batch_size - 1)
                else:
                    states_norm, actions_norm, rewards, dones, new_states_norm, __ = self.sample_batch(batch_size - 1)
                    # states = np.r_[states, old_state.reshape([1, 3])]
                    states_norm = np.r_[states_norm, old_state_norm.reshape([1, 2])]
                    actions_norm = np.r_[actions_norm, a_norm.reshape([1,1])]
                    rewards = np.r_[rewards, r.reshape([1,1])]
                    dones = np.r_[dones, done]
                    # new_states = np.r_[new_states, new_state.reshape([1,3])]
                    new_states_norm = np.r_[new_states_norm, new_state_norm.reshape([1,2])]
                    self.memorize(old_state_norm, a_norm, r, done, new_state_norm)

                new_states = new_states_norm * state_range + state_bias

                if t == t_max - 1:
                    # Select action according to the myopic policy
                    r_Ts = []

                    for new_state_T_1 in new_states:
                        r_T, a_T = env.myopic_policy(new_state_T_1)
                        r_Ts.append(r_T)
                    r_Ts = np.expand_dims(r_Ts, axis=1)
                    critic_target = rewards + self.gamma * r_Ts
                else:
                    # Predict target q-values using target networks
                    # Input to critic: (s_{t+1},a_{t+1})，Output: Q_value
                    q_values = self.critic.target_predict([new_states_norm, self.actor.target_predict(new_states_norm)])
                    # Compute critic target
                    critic_target = self.bellman(rewards, q_values, dones)

                    # Export results for Tensorboard
                    q = tfSummary('q-value', q_values[0])
                    summary_writer.add_summary(q, global_step=e)
                    summary_writer.flush()

                # Train both networks on sampled batch, update target networks
                self.update_critic(states_norm, actions_norm, critic_target)
                self.update_actor(states_norm, actions_norm)
                if e > 100:
                    self.epsilon *= self.decay
                # cumul_reward += r
                cumul_reward = critic_target[-1]

                # Export results for Tensorboard
                score = tfSummary('score', cumul_reward)
                summary_writer.add_summary(score, global_step=e)

                print("Score: " + str(cumul_reward))

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
