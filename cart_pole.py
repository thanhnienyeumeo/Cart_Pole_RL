import gym
import numpy as np
from gym.envs.registration import register
import matplotlib.pyplot as plt

from FrozenLake import do
env = gym.make('CartPole-v0')
# print(env.observation_space)
# print(env.action_space)
a = env.reset()[0]
print(a)
a  = np.reshape(a, [1, 4])
print(a)
# print(env.step(0))
done = False
# while not done:
#     env.render()
#     action = env.action_space.sample()
#     _, _, done, _, _ = env.step(action)

# do(env, 0.1, 0.99, 1000)