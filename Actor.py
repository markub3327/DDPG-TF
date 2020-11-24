from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, GaussianNoise
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf

# Trieda hraca
class Actor:

    def __init__(self, state_shape, action_shape, lr):
        state_input = Input(shape=state_shape)
        i = Dense(400, activation='relu', use_bias=True, kernel_initializer='he_uniform')(state_input)
        i = Dense(300, activation='relu', use_bias=True, kernel_initializer='he_uniform')(i)
        
        # vystupna vrstva   -- musi byt tanh pre (-1,1) ako posledna vrstva!!!
        output = Dense(action_shape[0], activation='tanh', use_bias=True, kernel_initializer='glorot_uniform')(i)

        # Vytvor model
        self.model = Model(inputs=state_input, outputs=output)

        # Skompiluj model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=self.optimizer, loss='mse')

        self.model.summary()

    @tf.function
    def train(self, X_train, criticNet):
        with tf.GradientTape() as tape:
            y_pred = self.model(X_train)
            q_pred = criticNet([X_train, y_pred])
        critic_grads = tape.gradient(q_pred, y_pred)

        # gradiendy siete
        with tf.GradientTape() as tape:
            y_pred = self.model(X_train, training=True)
        actor_grads = tape.gradient(y_pred, self.model.trainable_variables, output_gradients=-critic_grads)
        self.optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))
        #print(actor_grads)

    def save(self):
        plot_model(self.model, to_file='model_A.png')

