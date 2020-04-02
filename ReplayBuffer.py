from collections import deque
import numpy as np
import random

class ReplayBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)     #  Obojsmerne zretazena fronta

    def add(self, item):
        self.buffer.append(item)    # Ak je uz fronta plna vymaze prvy prvok zlava

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        item_count = len(batch[0])
        res = []
        for i in range(item_count):
            k = np.array([item[i] for item in batch])
            if len(k.shape)==1: k = k.reshape(k.shape+(1,))
            res.append(k)
        return res 

    def __len__(self):
        return len(self.buffer)

# Polozka v ReplayBuffer
class ReplayBufferItem:
    def __init__(self, state, action, reward, next_state, done):
        # informacie o odmene 'reward' za prechod do 'next_state' zo 'state' pouzitim 'action'
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __str__(self):
        return "ReplayBufferItem"