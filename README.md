# Deep Deterministic Policy Gradient (DDPG)

[![Release](https://img.shields.io/github/release/markub3327/MountainCarContinuous)](https://github.com/markub3327/MountainCarContinuous/releases)
![Tag](https://img.shields.io/github/v/tag/markub3327/MountainCarContinuous)

[![Issues](https://img.shields.io/github/issues/markub3327/MountainCarContinuous)](https://github.com/markub3327/MountainCarContinuous/issues)
![Commits](https://img.shields.io/github/commit-activity/w/markub3327/MountainCarContinuous)

![Languages](https://img.shields.io/github/languages/count/markub3327/MountainCarContinuous)
![Size](https://img.shields.io/github/repo-size/markub3327/MountainCarContinuous)

## Theory

&emsp;**Agent** is using DDPG algorithm to predict continuous actions in continuous state space. It has two networks: Actor and Critic.

https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
<br><br>
https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
<br><br>
https://spinningup.openai.com/en/latest/algorithms/ddpg.html

<p align="center"><b>Actor topology</b></p>
<p align="center">
  <img src="model_A.png" alt="Actor">
</p>

<p align="center"><b>Critic topology</b></p>
<p align="center">
  <img src="model_C.png" alt="Critic">
</p>

## Inputs/Outputs

&emsp;The Actor network has 2 inputs from game: position, velocity. The output layer consists from fully-connected 'tanh()' layer for doing actions in range (-1.0, 1.0): force. Hidden layers are using ReLU function.

&emsp;The Critic network has 2 inputs from game (states) and 1 input from Actor network (action). Hidden layers are using ReLU function. The main function of this network is estimate quality of the action[t] in the state[t].

The Critic network is trained by Bellman equation:
    
    Q_target = reward + (1-done) * gamma * Q_next_state

    Q_target       ->  Q value to be trained,
    reward         ->  reward from game for action in state,
    gamma          ->  discount factor,
    Q_next_state   ->  quality of action in next state 
    done           ->  1, if it's terminal state or 0 in non-terminal state

<p align="center"><b>Summary</b></p>
<p align="center">
  <img src="result.png" alt="Critic">
</p>

**Framework:** Tensorflow 2.0
</br>
**Languages:** Python 3 
</br>
**Author**: Martin Kubovcik
