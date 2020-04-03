from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numpy as np

# Trieda hraca
class Critic:

    def __init__(self, state_shape, action_shape, lr=0.001):
        # vstupna vsrtva pre state
        state_input = Input(shape=state_shape)
        state_h1 = Dense(48, activation='elu', use_bias=True, kernel_initializer='he_uniform')(state_input)

        # vstupna vrstva pre action
        action_input = Input(shape=action_shape)

        # equivalent to added = keras.layers.add([x1, x2])
        merged = Concatenate()([state_h1, action_input])
        merged_h1 = Dense(48, activation='elu', use_bias=True, kernel_initializer='he_uniform')(merged)
        merged_h2 = Dense(24, activation='elu', use_bias=True, kernel_initializer='he_uniform')(merged_h1)

        # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
        output = Dense(1, activation='linear', use_bias=True, kernel_initializer='he_uniform')(merged_h2)

        # Vytvor a skompiluj model
        self.model = Model(inputs=[state_input, action_input], outputs=output)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=self.optimizer, loss='mse')
                
        #self.frozen_model.summary()
        self.model.summary()

    def save(self):
        plot_model(self.model, to_file='model_C.png')


