import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, act_bias, lr, tau, seed):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.act_bias = act_bias
        self.tau = tau
        self.lr = lr
        self.seed = seed
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()
        

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=(self.env_dim,))
#        x = BatchNormalization()(inp)
        #
        # x = Dense(256, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(256), maxval = 1/np.sqrt(256), seed = self.seed))(inp)
        x = Dense(400, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(400), maxval = 1/np.sqrt(400), seed = self.seed))(inp)
        x = GaussianNoise(1)(x)
#        x = BatchNormalization()(x)
        #
        #x = Flatten()(x)
        # x = Dense(128, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(128), maxval = 1/np.sqrt(128), seed = self.seed))(x)
        x = Dense(300, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(300), maxval = 1/np.sqrt(300), seed = self.seed))(x)
        x = GaussianNoise(1)(x)
#        x = BatchNormalization()(x)

        # x = Dense(64, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(64), maxval = 1/np.sqrt(64), seed = self.seed))(x)
        x = Dense(100, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(100), maxval = 1/np.sqrt(100), seed = self.seed))(x)
        x = GaussianNoise(1)(x)

        out = Dense(self.act_dim, activation='tanh', kernel_initializer=RandomUniform(minval = -0.003, maxval = 0.003, seed = self.seed))(x)

        # out = Lambda(lambda i: i * self.act_range + self.act_bias)(out)
    
        #
        return Model(inp, out)


    # def network(self):
    #     """ Actor Network for Policy function Approximation, using a tanh
    #     activation for continuous control. We add parameter noise to encourage
    #     exploration, and balance it with Layer Normalization.
    #     """
    #     inp = Input(shape=(self.env_dim,))
    #     x = BatchNormalization()(inp)
    #     #
    #     x = Dense(256, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(256), maxval = 1/np.sqrt(256)))(x)
    #     x = GaussianNoise(1)(x)
    #     x = BatchNormalization()(x)
    #     #
    #     #x = Flatten()(x)
    #     x = Dense(128, activation='relu', kernel_initializer=RandomUniform(minval = -1/np.sqrt(128), maxval = 1/np.sqrt(128)))(x)
    #     x = GaussianNoise(1)(x)
    #     x = BatchNormalization()(x)

    #     out = Dense(self.act_dim, activation='tanh', kernel_initializer=RandomUniform(minval = -0.003, maxval = 0.003))(x)

 
    #     out = Lambda(lambda i: i * self.act_range + self.act_bias)(out)
    #     #
    #     return Model(inp, out)

    def predict(self, state):
        """ Action prediction
        """
        # np.expand_dims expand the shape of state as the input of actor
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model
        """
        # W, target_W = self.model.get_weights(), self.target_model.get_weights()
        # for i in range(len(W)):
        #     target_W[i] = W[i]
        # self.target_model.set_weights(target_W)

        W = self.model.get_weights()
        self.target_model.set_weights(W)

    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function(inputs=[self.model.input, action_gdts], outputs=[], updates=[tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
