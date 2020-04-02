from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numpy as np

# Trieda hraca
class Critic:

    def __init__(self, state_shape, action_shape):
        # vstupna vsrtva pre state
        state_input = Input(shape=state_shape)
        state_h1 = Dense(256, activation='elu', use_bias=True)(state_input)

        # vstupna vrstva pre action
        action_input = Input(shape=action_shape)

        # equivalent to added = keras.layers.add([x1, x2])
        merged = Concatenate()([state_h1, action_input])
        merged_h1 = Dense(128, activation='elu', use_bias=True)(merged)

        # vystupna vrstva
        output = Dense(1, activation='linear', use_bias=True)(merged_h1)

        # Vytvor a skompiluj model
        self.model = Model(inputs=[state_input, action_input], outputs=output)
        self.optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=self.optimizer, loss='mse')
                
        #self.frozen_model.summary()
        self.model.summary()

    def save(self):
        plot_model(self.model, to_file='model_C.png')


