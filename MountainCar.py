import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
import seaborn as sb

env = gym.make('MountainCar-v0')

action_space_size = env.action_space.n
pos_space = np.linspace(env.low[0], env.high[0], 20)
vel_space = np.linspace(env.low[1], env.high[1], 20)


def get_state(observation):
    pos, vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
    return pos_bin, vel_bin


Q_table = np.zeros((20, 20, action_space_size))
max_episode = 10000
env._max_episode_steps = 10000
epsilon = 1
learning_rate = 0.01
discount_rate = 0.99

reward_all_episodes = []
for episode in range(max_episode):

    state = env.reset()
    done = False
    reward_current_episode = 0

    for step in range(1000):
        current_pos = get_state(state)[0]
        current_vel = get_state(state)[1]

        explorateion_rate = random.uniform(0, 1)
        if explorateion_rate > epsilon:  # Exploitation
            action = np.argmax(Q_table[current_pos, current_vel, :])
        else:
            action = np.random.choice([0, 1, 2])  # Exploration

        new_state, reward, done, info = env.step(action)
        new_pos = get_state(new_state)[0]
        new_vel = get_state(new_state)[1]

        Q_table[current_pos, current_vel, action] = Q_table[current_pos, current_vel, action] * (
                1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(Q_table[new_pos, new_vel, :]))

        state = new_state
        reward_current_episode += reward

        if done:
            break

    if epsilon >= 0.001:
        epsilon -= np.exp(-0.001 * episode)

    reward_all_episodes.append(reward_current_episode)

reward_per_hundred_episodes = np.split(np.array(reward_all_episodes), max_episode / 1000)
count = 1000
print("**************Average reward per hundred episods************")
for r in reward_per_hundred_episodes:
    print(count, ":", str(sum(r / 1000)))
    count += 1000

for episode in range(3):
    state = env.reset()
    done = False
    print("*******EPISODE ", episode + 1, "*********\n\n\n\n")
    time.sleep(1)

    for step in range(1000):
        env.render()
        current_pos = get_state(state)[0]
        current_vel = get_state(state)[1]
        action = np.argmax(Q_table[current_pos, current_vel, :])
        new_state, reward, done, info = env.step(action)

        if done:
            time.sleep(3)
            break
        state = new_state

env.close()

#Visualization
solution_policy = np.max(Q_table, axis=2)
heat_map = sb.heatmap(solution_policy, annot=False)
plt.xlabel('Velocity')
plt.ylabel('Position')
poslist = np.linspace(env.low[0], env.high[0], 20).tolist()
poslist2 = [str(format(litem, '.2f')) for litem in poslist]
vellist = np.linspace(env.low[1], env.high[1], 20).tolist()
vellist2 = [str(format(litem, '.2f')) for litem in vellist]
heat_map.set_xticklabels(vellist2, rotation=45)
heat_map.set_yticklabels(poslist2, rotation=45)
plt.show()


plt.savefig('rl.png')
