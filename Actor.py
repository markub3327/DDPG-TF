from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, GaussianNoise
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf

# Trieda hraca
class Actor:

    def __init__(self, state_space, lr=0.0001):
        state_input = Input(shape=state_space)
        i = Dense(48, activation='elu', use_bias=True, kernel_initializer='he_uniform')(state_input)
        i = Dense(48, activation='elu', use_bias=True, kernel_initializer='he_uniform')(i)
        i = Dense(24, activation='elu', use_bias=True, kernel_initializer='he_uniform')(i)
        
        # vystupna vrstva   -- musi byt tanh pre (-1,1) ako posledna vrstva!!!
        out = Dense(1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform')(i)

        # Vytvor model
        self.model = Model(inputs=state_input, outputs=out)

        # Skompiluj model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=self.optimizer, loss='mse')

        self.model.summary()

    def train(self, X_train, critic_grads):
        # gradiendy siete
        with tf.GradientTape() as tape:
            y_pred = self.model(X_train, training=True)
        actor_grads = tape.gradient(y_pred, self.model.trainable_variables, output_gradients=-critic_grads)
        self.optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))
        #print(actor_grads)

    def save(self):
        plot_model(self.model, to_file='model_A.png')

