import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten

class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau, act_range, act_bias, seed, env):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.hisStep = env.hisStep
        self.tau, self.lr = tau, lr
        self.act_range, self. act_bias = act_range, act_bias
        self.seed = seed
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input(shape=(self.hisStep, self.env_dim))
        #        x = BatchNormalization()(state)
        action = Input(shape=(self.act_dim,))
        #        xa = Lambda(lambda i: (i - self.act_bias) / self.act_range)(action)
        x = LSTM(128, activation='relu',kernel_initializer=RandomUniform(minval=-1 / np.sqrt(128), maxval=1 / np.sqrt(128), seed=self.seed))(state)
        #        x = BatchNormalization()(x)
        #    x = concatenate([Flatten()(x), action])
        x = concatenate([x, action])
        x = Dense(128, activation='relu',kernel_initializer=RandomUniform(minval=-1 / np.sqrt(128), maxval=1 / np.sqrt(128), seed=self.seed))(x)
        x = Dense(64, activation='relu',kernel_initializer=RandomUniform(minval=-1 / np.sqrt(64), maxval=1 / np.sqrt(64), seed=self.seed))(x)

        out = Dense(1, activation='linear',kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003, seed=self.seed))(x)
        return Model([state, action], out)


    # def network(self):
        

    #     state = Input(shape=(self.env_dim,))
    #     x = BatchNormalization()(state)
    #     action = Input(shape=(self.act_dim,))
    #     xa = Lambda(lambda i: (i - self.act_bias) / self.act_range)(action)
    #     x = Dense(256, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(256), maxval = 1/np.sqrt(256)))(x)
    #     x = BatchNormalization()(x)
    #     #    x = concatenate([Flatten()(x), action])
    #     x = concatenate([x, xa])
    #     x = Dense(128, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(128), maxval = 1/np.sqrt(128)))(x)
            
    #     out = Dense(1, activation='linear', kernel_initializer=RandomUniform(minval = -0.003, maxval = 0.003))(x)
    #     return Model([state, action], out)

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    # def transfer_weights(self):
    #     """ Transfer model weights to target model with a factor of Tau
    #     """
    #     W, target_W = self.model.get_weights(), self.target_model.get_weights()
    #     for i in range(len(W)):
    #         target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
    #     self.target_model.set_weights(target_W)

    def transfer_weights(self):
        """ Transfer model weights to target model
        """
        # W, target_W = self.model.get_weights(), self.target_model.get_weights()
        # for i in range(len(W)):
        #     target_W[i] = W[i]
        # self.target_model.set_weights(target_W)

        W = self.model.get_weights()
        self.target_model.set_weights(W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
