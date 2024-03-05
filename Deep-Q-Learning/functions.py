# -*- coding: utf-8 -*-
"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

The class developed in this file implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
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
# import the necessary libraries
import numpy as np
import random
import torch
from collections import deque 
from tensorflow import gather_nd

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, output_dim)
    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
class DeepQLearning:
    
    ###########################################################################
    #   START - __init__ function
    ###########################################################################
    # INPUTS: 
    # env - Cart Pole environment
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes
    
            
    def __init__(self,env,gamma,epsilon,numberEpisodes):
        
        self.env=env
        self.gamma=gamma
        self.epsilon=epsilon
        self.numberEpisodes=numberEpisodes
        
        # state dimension
        self.stateDimension=4
        # action dimension
        self.actionDimension=2
        # this is the maximum size of the replay buffer
        self.replayBufferSize=300
        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batchReplayBufferSize=100
        
        # number of training episodes it takes to update the target network parameters
        # that is, every updateTargetNetworkPeriod we update the target network parameters
        self.updateTargetNetworkPeriod=100
        
        # this is the counter for updating the target network 
        # if this counter exceeds (updateTargetNetworkPeriod-1) we update the network 
        # parameters and reset the counter to zero, this process is repeated until the end of the training process
        self.counterUpdateTargetNetwork=0
        
        # this sum is used to store the sum of rewards obtained during each training episode
        self.sumRewardsEpisode=[]
        
        # replay buffer
        self.replayBuffer=deque(maxlen=self.replayBufferSize)
        
        # this is the main network
        # create network
        self.mainNetwork=DQN(self.stateDimension, self.actionDimension)
        
        # this is the target network
        # create network
        self.targetNetwork=DQN(self.stateDimension, self.actionDimension)
        
        # copy the initial weights to targetNetwork
        self.targetNetwork.load_state_dict(self.mainNetwork.state_dict())
        
        # this list is used in the cost function to select certain entries of the 
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend=[]
    
        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.mainNetwork.parameters(), lr=0.01)
        self.epochs = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###########################################################################
    #   END - __init__ function
    ###########################################################################
    
    ###########################################################################
    #   START - function trainingEpisodes()
        

    def trainingEpisodes(self):
   
        
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):
            
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewardsEpisode=[]
                       
            print("Simulating episode {}".format(indexEpisode))
            
            # reset the environment at the beginning of every episode
            (currentState,_)=self.env.reset()
                      
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState=False
            while not terminalState:
                                      
                # select an action on the basis of the current state, denoted by currentState
                action = self.selectAction(currentState,indexEpisode)
                
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                (nextState, reward, terminalState,_,_) = self.env.step(action)          
                rewardsEpisode.append(reward)
         
                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replayBuffer.append((currentState,action,reward,nextState,terminalState))
                
                # train network
                self.trainNetwork()
                
                # set the current state for the next step
                currentState=nextState
            
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
    ###########################################################################
    #   END - function trainingEpisodes()
    ###########################################################################
            
       
    ###########################################################################
    #    START - function for selecting an action: epsilon-greedy approach
    ###########################################################################
    # this function selects an action on the basis of the current state 
    # INPUTS: 
    # state - state for which to compute the action
    # index - index of the current episode
    def selectAction(self,state,index):
        import numpy as np
        
        # first index episodes we select completely random actions to have enough exploration
        # change this
        if index<1:
            return np.random.choice(self.actionDimension)   
            
        # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber=np.random.random()
        
        # after index episodes, we slowly start to decrease the epsilon parameter
        if index>200:
            self.epsilon=0.999*self.epsilon
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionDimension)            
        
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qvalues[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
                       
            Qvalues=self.mainNetwork(torch.Tensor(state.reshape(1,4))).squeeze()
            
           # print(np.max(Qvalues))
            return torch.where(Qvalues==torch.max(Qvalues))[0].item()
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example 
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple
    ###########################################################################
    #    END - function selecting an action: epsilon-greedy approach
    ###########################################################################
    
    #    START - function trainNetwork() - this function trains the network 
    def trainNetwork(self):

        # if the replay buffer has at least batchReplayBufferSize elements,
        # then train the model 
        # otherwise wait until the size of the elements exceeds batchReplayBufferSize
        if (len(self.replayBuffer)>self.batchReplayBufferSize):
            

            # sample a batch from the replay buffer
            randomSampleBatch=random.sample(self.replayBuffer, self.batchReplayBufferSize) #n x [state,action,reward,nextState,terminated]
            
            # here we form current state batch 
            # and next state batch
            # they are used as inputs for prediction
            currentStateBatch=np.zeros(shape=(self.batchReplayBufferSize,4))
            nextStateBatch=np.zeros(shape=(self.batchReplayBufferSize,4))            
            # this will enumerate the tuple entries of the randomSampleBatch
            # index will loop through the number of tuples
            for index,tupleS in enumerate(randomSampleBatch):
                # first entry of the tuple is the current state
                currentStateBatch[index,:]=tupleS[0] #n x state
                # fourth entry of the tuple is the next state
                nextStateBatch[index,:]=tupleS[3] # n x nextState
            
            # here, use the target network to predict Q-values 
            QnextStateTargetNetwork=self.targetNetwork(torch.Tensor(nextStateBatch))
            # print(QnextStateTargetNetwork)
            # now, we form batches for training
            # input for training
            inputNetwork=currentStateBatch #nxstate
            # output for training
            outputNetwork=np.zeros(shape=(self.batchReplayBufferSize,2))
            
            
            # here, we train the network
            # self.mainNetwork.fit(inputNetwork,outputNetwork,batch_size = self.batchReplayBufferSize, verbose=0,epochs=100)
            y_batch, y_target_batch = [], []     
            for index, (currentState,action,reward,nextState,terminated) in enumerate(randomSampleBatch):
                reward = torch.tensor(reward)
                currentState = torch.tensor(currentState, dtype=torch.float32)
                y = self.mainNetwork(currentState).squeeze()
                y_target = y.clone().detach()
                with torch.no_grad():
                    y_target[action] = reward if terminated else reward + self.gamma * torch.max(QnextStateTargetNetwork[index])
                y_target_batch.append(y_target)
                y_batch.append(y)
            y_batch = torch.cat(y_batch)
            y_target_batch = torch.cat(y_target_batch)
            self.opt.zero_grad()
            loss = self.criterion(y_batch, y_target_batch)
            loss.backward()
            self.opt.step()
            # after updateTargetNetworkPeriod training sessions, update the coefficients 
            # of the target network
            # increase the counter for training the target network
            self.counterUpdateTargetNetwork+=1  
            if (self.counterUpdateTargetNetwork>(self.updateTargetNetworkPeriod-1)):
                # copy the weights to targetNetwork
                self.targetNetwork.load_state_dict(self.mainNetwork.state_dict())        
                print("Target network updated!")
                # reset the counter
                self.counterUpdateTargetNetwork=0
                torch.save(self.mainNetwork.state_dict(), "model.pth")
    ###########################################################################
    #    END - function trainNetwork() 
    ###########################################################################     
                  
                
                
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
    