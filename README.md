# DDPG Actor-Critic Reinforcement Learning Reacher Environment

## Summary
This is the repository for my trained Deep Deterministic Policy Gradient based agent on the Unity Reacher Enviroment from the Deep Reinforcement Learning nanodegree program. To 'solve' the environment the agent must navigate the Envirnoment with an average score of greater than 30 over the last 100 episodes. This repository provides the code to achieve this in 110 episodes. 



## Environment
This Unity environment requires an agent to control 20 identical robot arms to position them in moving target loctions.

<img src="https://github.com/Remtasya/DDPG-Actor-Critic-Reinforcement-Learning-Reacher-Environment/blob/master/project_images/reacher environment.gif" alt="Environment" width="700"/>

The task is episodic with termination after 1000 timesteps.


### State space
A state is represented by a vector of 33 dimensions, which contains information about the agent and environment.

### Action space
An action consists of a 4 diminsional vector with values between -1 and 1, which corresponds to forces applied to the joints of the robotic arm.

### Reward
Positioning the arm inside the moving target provides a reward of 0.1 per timestep.


## Dependencies
In order to run this code you will require:

1.  Python 3 with the packages in the following repository: https://github.com/udacity/deep-reinforcement-learning, including pytorch.

2.  The ml-agents package, which can be the installed following the following instructions: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

3.  The Reacher Unity environment specific to your operating system, which can be found here: https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control. After cloning this environment download the Reacher environment appropriate to your operating system, place the Reacher Folder with the root directory, and change it's path when loaded at the beginning of the notebooks.


## How to run the repository


### How to watch a random agent
To confirm the environment is set up correctly I recommend running the random_agent.ipynb notebook to observe a randomly-acting agent.

### How to train an agent
To run the code from scratch simply open the train_agent.ipynb notebook and run the code.

### How to test a trained agent
To test a pre-trained agent (I've included my trained one in this repository) simply open the test_agent.ipynb notebook and run the code.


## What files are included

### ipynb files
As stated above train_agent.ipynb and test_agent.ipynb are intuitive files that are all that's required to walk you through training or testing this agent. If however you would like to change the code (such as to specify a different model architecture, or hyperparameter selection) then you may find the following descriptions useful:

### report.md
This describes additional details of the implementation such as the empirical results, hyperparameter selections and other specifics of the DDPG algorithm.

### model.py
This is a simple python script that specifies the pytorch model architectures used for the Actor Network and Critic Network. For this project the architecture is quite straightforward, simple feed-forward neural networks with linear layers. 

### DDPG_agent.py
This file contains all of the functions required for the agent to store experience, sample and learn from it, and select actions in the enviroment.

### checkpoint.pth
This file contains the trained weights of the most recently trained agent. You can use this file to straight away test an agent without having to train one yourself.


## Agent design and implementation


**Details of the agent design can also be found in the Report.md, but a summary is provided here:**

The algorithm used is based on the Deep Reinforcement Learning DDPG algorithm described in this paper: https://arxiv.org/pdf/1509.02971.pdf

\
Deep Reinforcement Learning is an innovative approach that effectively combines two seperate fields:

### Reinforcement Learning
In Reinforcement learning, the goal is to have an agent learn how to navigate a new enviroment with the goal of maximising cummulative rewards. One approach to this end is Q-learning, where the agent tries to learn the dynamics of the enviroment indirectly by focusing on estimating the value of each state-action pair in the enviroment. This is acheived over the course of training, using it's experiences to produce and improve these estimates - as the agent encounters state-action pairs more often it becomes more confident in its estimate of their value. 

### Deep Learning
Famous in computer vision and natural language processing, deep learning uses machine learning to make predictions by leveraging vast amounts of training data and a flexible architecture that is able to generalise to previously unseen examples. In Deep Reinforcement Learning we leverage this power to learn which actions to take, and use the agents experiences within the enviroment as a reusable form of training data. This proves to be a powerful combination thanks to Deep learning's ability to generalise given sufficent data and flexibility.

\
**Combined these two fields lead to:**

### Deep Q Learning
The Q network is designed to map state-action combinations to values. Thus we can feed it our current state and then determine the best action as the one that has the largest estimated state-action value. In practice we typically adopt a somewhat random action early on to encourage initial exporation. After we've collected enough state-action-reward-state experiences we start updating the model. This is acheived by sampling some of our experiences and then computing the empirically observed estimates of the state-action values compared to those estimated from the model. The difference between these two is coined the TD-error and we then make a small modification to the model weights to reduce this error, via neural network backpropagation of the TD-error. We simply iterate this process over many timesteps per episode, and many episodes, until convergence of the model weights is acheived.

### Actor-Critic methods
Deep Q-Learning is designed for environments with discreet action spaces, but struggles to generalise to continuous action spaces. This is because Q-Learning works by computing all possible state-actions values and then choosing the highest one. In continuous action spaces Q-value iteration is not possible and instead would require a continuous optimisation to select best actions. Actor-Critic methods rememdy this problem by using two networks instead of just the Q-network:

The **Critic Network** is a modified Q-network estimator that is designed to output the value for any given state-action value rather than iterating through all of them. As in Q-Learning, it uses the Bellman equations for learning the Q-values.

The **Actor network** attempts to estimate the optimal action directly given a state. It is learned by making use of the Critic Network as a baseline, and is used for selecting which actions to take.

### Deep Deterministic Policy Gradients
In addition to using an actor-critic setup, the Deep Deterministic Policy Gradient algorithm (https://arxiv.org/pdf/1509.02971.pdf) additionally makes use of the success of Deep Q-networks to incorporate off-policy learning by use of a replay buffer and target networks for both the Critic and Actor which are updated periodically, and these modification enable more stable learning.

## Further additions
Additions that might improve the algorithm further are those of the D4PG algorithm, which inculdes prioritised experience replay, distributional value-learning, and n-step bootstrapping.


**Namely these are:**

1.  Prioritised Experience Replay - https://arxiv.org/abs/1511.05952
2.  Distributional DQN - https://arxiv.org/abs/1707.06887
3.  Learning from multi-step bootstrap targets -  https://arxiv.org/abs/1602.01783

