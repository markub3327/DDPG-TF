import gym
import matplotlib.pyplot as plt
import numpy as np
import Actor
import Critic
import ReplayBuffer
import tensorflow as tf
import wandb

from wandb.keras import WandbCallback
from noise import OrnsteinUhlenbeckActionNoise

# inicializuj prostredie Weights & Biases
wandb.init(project="stable-baselines")

wandb.config.gamma = 0.98
wandb.config.batch_size = 100
wandb.config.tau = 0.005
wandb.config.lr_A=0.001
wandb.config.lr_C=0.001
wandb.config.learning_start = 100

# Herne prostredie
env = gym.make('MountainCarContinuous-v0')

# Actor
actorNet = Actor.Actor(env.observation_space.shape, env.action_space.shape, lr=wandb.config.lr_A)
actorNet_target = Actor.Actor(env.observation_space.shape, env.action_space.shape, lr=wandb.config.lr_A)

# Critic
criticNet = Critic.Critic(env.observation_space.shape, env.action_space.shape, lr=wandb.config.lr_C)
criticNet_target = Critic.Critic(env.observation_space.shape, env.action_space.shape, lr=wandb.config.lr_C)

# replay buffer
rpm = ReplayBuffer.ReplayBuffer(1000000) # 1M history

noise = OrnsteinUhlenbeckActionNoise(mean=0.0, sigma=0.5, size=env.action_space.shape)

# (gradually) replace target network weights with online network weights
def replace_weights(tau=wandb.config.tau):
    theta_a,theta_c = actorNet.model.get_weights(),criticNet.model.get_weights()
    theta_a_targ,theta_c_targ = actorNet_target.model.get_weights(),criticNet_target.model.get_weights()

    # mixing factor tau : we gradually shift the weights...
    theta_a_targ = [theta_a[i]*tau + theta_a_targ[i]*(1-tau) for i in range(len(theta_a))]
    theta_c_targ = [theta_c[i]*tau + theta_c_targ[i]*(1-tau) for i in range(len(theta_c))]

    actorNet_target.model.set_weights(theta_a_targ)
    criticNet_target.model.set_weights(theta_c_targ)

def train(verbose=1, batch_size=wandb.config.batch_size, gamma=wandb.config.gamma):
    # ak je dostatok vzorov k uceniu
    if (len(rpm) > batch_size):        
        [s1, a1, r, s2, done] = rpm.sample(batch_size)
        #print(s1.shape, a1.shape, r.shape, done.shape, s2.shape)

        # ---------------------------- update critic ---------------------------- #
        # a2_targ = actor_targ(s2) : what will you do in s2, Mr. old actor?
        a2 = actorNet_target.model(s2)
        #print(s2)

        # q2_targ = critic_targ(s2,a2) : how good is action a2, Mr. old critic?
        q2 = criticNet_target.model([s2, a2])

        # Use Bellman Equation! (recursive definition of q-values)
        q1_target = r + (1-done) * gamma * q2
        #print(q1_target)

        #print("Critic network")
        criticNet.model.fit([s1, a1], q1_target, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=False, callbacks=[WandbCallback()])
        # ---------------------------- update actor ---------------------------- #
        #print("Actor network")
        actorNet.train(s1, criticNet.model)
        
        replace_weights()

def main():
    # list skore
    scoreList = []

    # prekopiruj vahy Critic a Actor do Critic_Target a Actor_Target
    replace_weights(tau=1.)

    # interacia epizod
    for episode in range(500):
        # stav z hry
        state = env.reset()
        state = np.reshape(state, (1, env.observation_space.shape[0]))

        noise.reset()

        # kroky v hre (epizody)
        score = 0.0
        for step in range(1500):
            # prekresli okno hry
            env.render()

            if (len(rpm) < wandb.config.learning_start):
                action = env.action_space.sample()
            else:
                action = actorNet.model(state)[0]
                # Clip continuous values to be valid w.r.t. environment
                action = np.clip(action + noise(), -1.0, 1.0)
            
            # krok hry
            newState, reward, done, info = env.step(action) 
            newState = np.reshape(newState, (1, env.observation_space.shape[0]))

            if (step == 1):
                print(f"State: {state}")  # stav v hre
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                print(f"Is done? {done}\n")

            # sucet odmien
            score += reward

            # uloz nazbierane udaje z hry do fronty
            rpm.add((np.squeeze(state), action, reward, np.squeeze(newState), done))

            # prekopiruj novy stav
            state = newState

            # koniec epizody, uspesne dorazil do ciela
            if done: 
                print("Episode finished after {} timesteps\n".format(step+1))
                break

            # musi sa ucit za kazdou iteraciou
            if (len(rpm) >= wandb.config.learning_start):
                verbose = 1 if step == 1 else 0
                train(verbose)

        # Vypis skore a pridaj do listu
        #print(f"Epsilon: {noise_level}")
        wandb.log({"score":score})
        print(f"Score: {score}\n")
        print(f"Episode: {episode}\n")
        print(f"Steps: {step}")
        scoreList.append(score)

    # nastav graf
    plt.xlabel('time')
    plt.ylabel('score')
    plt.title("Score plot")
    #plt.legend()

    # plot the data itself
    plt.plot(scoreList, label='Score', color='red')

    # Uloz a zobraz graf skore
    plt.savefig('result.png')
    plt.show()

    actorNet.save()
    criticNet.save()

    # Save model to wandb
    actorNet.model.save(wandb.os.path.join(wandb.run.dir, "model_A.h5"))
    criticNet.model.save(wandb.os.path.join(wandb.run.dir, "model_C.h5"))

    # zatvor prostredie
    env.close()

if __name__ == "__main__":
    main()