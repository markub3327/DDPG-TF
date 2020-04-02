import gym
import matplotlib.pyplot as plt
import numpy as np
import Actor
import Critic
import ReplayBuffer
import tensorflow as tf

# Herne prostredie
env = gym.make('MountainCarContinuous-v0')

# Actor
actorNet = Actor.Actor(env.observation_space.shape)
actorNet_target = Actor.Actor(env.observation_space.shape)

# Critic
criticNet = Critic.Critic(env.observation_space.shape, env.action_space.shape)
criticNet_target = Critic.Critic(env.observation_space.shape, env.action_space.shape)

# replay buffer
rpm = ReplayBuffer.ReplayBuffer(1000000) # 1M history

class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=750, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x

# (gradually) replace target network weights with online network weights
def replace_weights(tau=0.01):
    theta_a,theta_c = actorNet.model.get_weights(),criticNet.model.get_weights()
    theta_a_targ,theta_c_targ = actorNet_target.model.get_weights(),criticNet_target.model.get_weights()

    # mixing factor tau : we gradually shift the weights...
    theta_a_targ = [theta_a[i]*tau + theta_a_targ[i]*(1-tau) for i in range(len(theta_a))]
    theta_c_targ = [theta_c[i]*tau + theta_c_targ[i]*(1-tau) for i in range(len(theta_c))]

    actorNet_target.model.set_weights(theta_a_targ)
    criticNet_target.model.set_weights(theta_c_targ)

def train(verbose, batch_size=128, gamma=0.95):
    # ak je dostatok vzorov k uceniu
    if (len(rpm) > batch_size):        
        [s1, a1, r, s2, done] = rpm.sample(batch_size)
        #print(s1.shape, a1.shape, r.shape, done.shape, s2.shape)

        # ---------------------------- update critic ---------------------------- #
        # a2_targ = actor_targ(s2) : what will you do in s2, Mr. old actor?
        a2 = actorNet_target.model.predict(s2)
        #print(s2)

        # q2_targ = critic_targ(s2,a2) : how good is action a2, Mr. old critic?
        q2 = criticNet_target.model.predict([s2, a2])

        # Use Bellman Equation! (recursive definition of q-values)
        q1_target = r + (1-done) * gamma * q2
        #print(q1_target)

        #print("Critic network")
        criticNet.model.fit([s1, a1], q1_target, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=False)
        # ---------------------------- update actor ---------------------------- #
        with tf.GradientTape() as tape:
            y_pred = actorNet.model(s1)
            q_pred = criticNet.model([s1, y_pred])
        critic_grads = tape.gradient(q_pred, y_pred)
        #print(critic_grads)

        #print("Actor network")
        actorNet.train(s1, critic_grads)
        
        replace_weights()

def clamper(actions):
    return np.clip(actions,a_max=env.action_space.high,a_min=env.action_space.low)

def main():
    # list skore
    scoreList = []

    # prekopiruj vahy Critic a Actor do Critic_Target a Actor_Target
    replace_weights(tau=1.)

    # interacia epizod
    for episode in range(1000):
        # stav z hry
        state = env.reset()
        state = np.reshape(state, (1, env.observation_space.shape[0]))

        noise = OrnsteinUhlenbeckProcess(size=env.action_space.shape)

        # kroky v hre (epizody)
        score = 0.0
        for step in range(200):
            # prekresli okno hry
            env.render()

            # add noise to our actions, since our policy by nature is deterministic
            exploration_noise = noise.generate(episode)
            
            action = np.squeeze(actorNet.model(state))
            # Clip continuous values to be valid w.r.t. environment
            action = np.clip(action+exploration_noise, -1.0, 1.0)
            
            # krok hry
            newState, reward, done, info = env.step(action) 
            newState = np.reshape(newState, (1, env.observation_space.shape[0]))

            if (step == 1):
                print("exploration_noise: {}".format(exploration_noise))
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
                print("Episode finished after {} timesteps".format(step+1))
                break

        #verbose = 1 if step == 1 else 0    
        train(1)#(verbose)

        # Vypis skore a pridaj do listu
        #print(f"Epsilon: {noise_level}")
        print(f"Score: {score}\n")
        print(f"Episode: {episode}\n")
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

    # zatvor prostredie
    env.close()

if __name__ == "__main__":
    main()