"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

This file contains driver code that imports DeepQLearning class developed in the file "functions_final"
 
The class DeepQLearning implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
The implementation is based on the OpenAI Gym Cart Pole environment and TensorFlow (Keras) machine learning library

The webpage explaining the codes and the main idea of the DQN is given here:

https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/


Author: Aleksandar Haber 
Date: February 2023

Tested on:

tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.11.0
tensorflow-estimator==2.11.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.30.0

keras==2.11.0

gym==0.26.2

"""
# import the class
import matplotlib.pyplot as plt
from torchsummary import summary
from functions import DeepQLearning
# classical gym 
import gym
import torch
# instead of gym, import gymnasium 
#import gymnasium as gym

# create environment
env=gym.make('CartPole-v1')

# select the parameters
gamma=1
# probability parameter for the epsilon-greedy approach
epsilon=0.1
# number of training episodes
# NOTE HERE THAT AFTER CERTAIN NUMBERS OF EPISODES, WHEN THE PARAMTERS ARE LEARNED
# THE EPISODE WILL BE LONG, AT THAT POINT YOU CAN STOP THE TRAINING PROCESS BY PRESSING CTRL+C
# DO NOT WORRY, THE PARAMETERS WILL BE MEMORIZED
numberEpisodes=10

# create an object
LearningQDeep=DeepQLearning(env,gamma,epsilon,numberEpisodes)
print(LearningQDeep.device)
# run the learning process
LearningQDeep.trainingEpisodes()
# get the obtained rewards in every episode
sumRewardsEpisode =  LearningQDeep.sumRewardsEpisode

#  summarize the model
summary(LearningQDeep.mainNetwork,(1,4))
plt.plot(sumRewardsEpisode, marker='o', linestyle='-')
plt.title('Biến động của Rewards qua các tập (episodes)')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.grid(True)  
plt.savefig("Deep-Q-Learning_rewards.png")
plt.show()

torch.save(LearningQDeep.mainNetwork.state_dict(), "model.pth")
# save the model, this is important, since it takes long time to train the model 
# and we will need model in another file to visualize the trained model performance
# LearningQDeep.mainNetwork.save("trained_model_temp.h5")



