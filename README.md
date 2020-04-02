# MountainCarContinuous

## Theory

**Agent** is using DDPG algorithm to predict continuous actions in continuous state space. It has two networks: Actor and Critic.

    grad_J = (dQ / daction) * (daction / dA)

    grad_J ->  policy gradient
    Q      ->  Q value from Critic net on (state, action) pair
    action ->  deterministic policy predicated by Actor
    A      ->  Actor's weights

**Framework:** Keras, Tensorflow 2.0
</br>
**Languages:** Python 3, 
</br>
**Author**: Martin Kubovcik