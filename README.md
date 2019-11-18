# Distributional Multi-agent DDPG Actor-Critic Reinforcement Learning Tennis Environment

## Summary
This is the repository for my trained Multi-Agent Deep Deterministic Policy Gradient based agent on the Unity Tennis Enviroment from the Deep Reinforcement Learning nanodegree program. To 'solve' the environment the agents must be able to obtain a 100-episode rolling average score of 0.5. This repository provides the code to achieve this in 2600 episodes, and in 3200 episodes is able to acheive a score of 1.5, which is comparable with expert level human play, as shown below:


<img src="https://github.com/Remtasya/MADDPG/blob/master/project_images/trained_agent.gif" alt="Environment" width="700"/>

**Note on Stabilty**
Reinforcement Learning is an exciting new area of machine learning with a lot of promise, but one downside of its ambitious goals and lack of maturity is that the algorithms tend to be far less stable than traditional machine learning - especially in Multi-agent settings such as this. This is exhibited in this project through great sensitivity to hyperparameters, large variation between runs with identical hyperparameters due by choatic feed-back loops, asymetric agent performance, and catastrophic forgetting.

## Environment
This Unity environment requires two agents controling rackets to rally a ball with eachother.
The environment is reset to a new episode if the agents drop the ball or if 1000 timesteps have elapsed - whichever comes first.

### State space
A state is represented by a vector of 24 dimensions for each agent, which contains information about the position and velocity of agent's racket and the ball. Thus overall for both agents we have a 48 dimension state-space.

### Action space
An action for each agent consists of a 2 diminsional vector with values between -1 and 1, which corresponds to the direction the racket is moved in. Thus overall for both agents we have a 4 dimension action-space.

### Reward
Successfully hitting the ball over the net yields a reward of 0.1 per timestep, while droping the ball or hitting it out of bounds results in a reward of -0.01 and the episode terminating.

## Dependencies
In order to run this code you will require:

1.  Python 3 with the packages in the following repository: https://github.com/udacity/deep-reinforcement-learning, including pytorch.

2.  The ml-agents package, which can be the installed following the following instructions: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

3.  The Tennis Unity environment specific to your operating system, which can be found here: https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet. From here download the tennis environment appropriate to your operating system, place the Tennis Folder within the root directory of this cloned repository, and change it's path if required when loaded at the beginning of the notebooks.

## How to run the repository

### How to watch a random agent
To confirm the environment is set up correctly I recommend running the random_agent.ipynb notebook to observe a pair of randomly-acting agents.

### How to train an agent
To run the code from scratch simply open the train_agent.ipynb notebook and run the code.

### How to test a trained agent
To test a pair of pre-trained agents (by default my trained ones are in this repository) simply open the test_agent.ipynb notebook and run the code. With no noise added to the actions, the agents are able to obtain an average score as high as 2.6!

## Included files and what they do

### ipynb files
As stated above train_agent.ipynb and test_agent.ipynb are intuitive files that are all that's required to walk you through training or testing this agent. If however you would like to change the code (such as to specify a different model architecture, or hyperparameter selection) then you may find the following descriptions useful:

### report.md
This describes additional details of the implementation such as the empirical results, hyperparameter selections and other specifics of the MADDPG algorithm.

### maddpg.py
This is a main script that specifies most of the hyperparameters and consolidates the various specific scripts that are required for the MADDPG algorithm, such as the ddpg agents, the neural networks for the actors and critics, and the pre-processing of the inputs and outputs from the environment. In addition it also handles the updating of the actors and critics of both agents, which are quite complex for MADDPG and distributional Q-learning and are best read about in those two papers, found in the glossary at the end. For the distributional Q-learning it also includes the to_categorical function which is used in the updating of the critic to transform the Q-values to a distribution before calculating cross-entropy.

### ddpg.py
This file contains all the initialisation for a single ddpg agent, such as it's actor and critic network as well as the target networks. It also defines the action step, where a state is fed into the network and an action combined with noise is produced. It also defines the target_action of the target actor, which is used in the update according to the Q-target procedure of DQN.

### network.py
This is a script that specifies the pytorch model architectures used for the Actor Network and Critic Networks and target networks, as well as weight initialisation and forward steps. The architecture is relatively straightforward, using feed-forward neural networks with linear layers.  The Actor network inputs a state and outputs an action, while the Critic network approximates a Q-value with an action-state pair as input. The critic has some extra complexity oweing to the Multi-Agent and distributional Q-value extensions: 

1.  Firstly it takes as inputs the states and actions of both agents, which is refered to in the MADDPG algorithm paper as centeralised training but decentralised execution, since the actors only use their own states.
2.  Secondly it uses the Distributional DQN approach to estimate a distribution of Q-values instead of just the expected Q-value. Thus it outputs a probability distribution over Q-value bins, and hence uses softmax activation.

### noise.py
Noise is critical to good performance of an RL agent as it determines how the trade-off between exploration and exploitation is managed. Too little noise can result in the agent getting stuck in local optima indefinitely during training, while too much noise can result in catastrophic forgetting of good learned behaviour. This script contains 4 types of noise to choose from: 
1.  OUNoise - which produces serially correlated noise with a mean-reverting property. Because it is serially correlated it can produce actions that are smoother in appearance than other types of noise - for example uniform noise will appear jittery by comparison.
2.  BetaNoise - which transforms the network action using a beta distribution. Because the beta distribution has a support from 0 to 1 it can easily be extended to -1 to 1, which makes it able to provide variance and mean adaptive noise in both directions without requiring clipping to the -1 to 1 interval.
3.  GaussNoise - which simply adds standard white noise which is then clipped to the -1 to 1 interval.
4.  WeightedNoise - which simply takes a weighted average of uniform noise on the range -1 to 1 with the action.
All of these noises differ in the extent to which they anneal their variance, and favour the center i.e. 0 of the interval versus respect the action value with their mean. They can easily be experimented with to determine the most suitable choice for an environment.
See below for an illustration of the beta distribution's abilty to adapt it's mean within the 0 to 1 support:

<img src="https://github.com/Remtasya/MADDPG/blob/master/project_images/beta.gif" alt="Environment" width="400"/>


### buffer.py
This file contains the replay buffer, which is a deque used for storing all of the experience collected by the agent. As per the experience replay procedure this is then randomly sampled from in minibatches to train the agent. Also, as per the 'learning from multi-step bootstrap targets' paper, we're utilising n-step returns, which means we examine the cause-and-effect of actions not over a single timestep but across several, which can help the agent learn when there is a delay between action and reward.

### utilities.py
contains various helper functions related to small steps which are utilised repeatedly in the rest of the code, such as hard and soft updates of the target networks, and transposing lists of tensors representing batches of experience collected by the agent.

### folder: model_dir
This folder contains the trained weights perodically of the agents. You can use these to test an agent without having to train one yourself, or to observe the behaviour of an agent you've trained yourself.

### folder: log
This folder contains metrics collected over the course of training to be viewed in tensorboard, such as the loss metrics of the actor and critic for both agents. Tensorboard is not strictly necessary for training, and a progressbar is also used in the notebook showing infomation such as performance metrics.


## Theory overview - from RL to MADDPG 

*More technical details of the agent design such as hyperparameters chosen can be found in the Report.md*


**Deep Reinforcement Learning is an innovative approach that effectively combines two seperate fields:**

### Reinforcement Learning
In Reinforcement learning, the goal is to have an agent learn how to navigate a new enviroment with the goal of maximising cummulative rewards. One approach to this end is Q-learning, where the agent tries to learn the dynamics of the enviroment indirectly by focusing on estimating the value of each state-action pair in the enviroment. This is acheived over the course of training, using it's experiences to produce and improve these estimates - as the agent encounters state-action pairs more often it becomes more confident in its estimate of their value. 

### Deep Learning
Famous in computer vision and natural language processing, deep learning uses machine learning to make predictions by leveraging vast amounts of training data and a flexible architecture that is able to generalise to previously unseen examples. In Deep Reinforcement Learning we leverage this power to learn which actions to take, and use the agents experiences within the enviroment as a reusable form of training data. This proves to be a powerful combination thanks to Deep learning's ability to generalise given sufficent data and flexibility.


**Combined these two fields lead to:**

### Deep Q Learning
The Q network is designed to map state-action combinations to values. Thus we can feed it our current state and then determine the best action as the one that has the largest estimated state-action value. In practice we typically adopt a somewhat random action early on to encourage initial exporation. After we've collected enough state-action-reward-state experiences we start updating the model. This is acheived by sampling some of our experiences and then computing the empirically observed estimates of the state-action values compared to those estimated from the model. The difference between these two is coined the TD-error and we then make a small modification to the model weights to reduce this error, via neural network backpropagation of the TD-error. We simply iterate this process over many timesteps per episode, and many episodes, until convergence of the model weights is acheived.

### Actor-Critic methods
Deep Q-Learning is designed for environments with discreet action spaces, but struggles to generalise to continuous action spaces. This is because Q-Learning works by computing all possible state-actions values and then choosing the highest one. In continuous action spaces Q-value iteration is not possible and instead would require a continuous optimisation to select best actions. Actor-Critic methods rememdy this problem by using two networks instead of just the Q-network:

The **Critic Network** is a modified Q-network estimator that is designed to output the value for any given state-action value rather than iterating through all of them. As in Q-Learning, it uses the Bellman equations for learning the Q-values.

The **Actor network** attempts to estimate the optimal action directly given a state, and is used for selecting which actions to take. It is learned by making use of the Critic Network as a baseline, where the critic evaulates how good the actor's action is.

### Deep Deterministic Policy Gradients
In addition to using an actor-critic setup, the Deep Deterministic Policy Gradient algorithm (https://arxiv.org/pdf/1509.02971.pdf) additionally makes use of the success of Deep Q-networks to incorporate off-policy learning by use of a replay buffer and target networks for both the Critic and Actor which are updated periodically, and these modification enable more stable learning.

### Multi-Agent Deep Deterministic Policy Gradients

A multi-agent setting introduces several additional complexities. For example, from the perspective of a single agent, it is tempting to view other agents as part of the environment,as if they merely receive an action and output a state response. Note however that because the other agents are updating over the course of training, their behaviour is changing. Thus off policy training - such as experience replay, which is crucial for stablising DDPG - would violate a key assumption of Reinforcement Learning of having a stationary environment. The MADDPG paper (https://arxiv.org/pdf/1706.02275.pdf) introduces the concept of centralised training but de-centralised execution as a work-around for this problem. Essentially each critic is augmented to receive as inputs the actions and states of all other agents as well as it's own, and during updates utilises other agent's actors and target actors rather than viewing them as part of the environment. This omniscient critic is used to train the actor, by performing gradient ascent on the critic's evaluation of the actor's action. During execution outside of training though the actor is able to act freely without access to other agent's internal models.

\
Note that MADDPG also includes some other additions, such as prioritised experience replay and policy ensembles, which were not used in this implementation. Also noteworthy is the fact that MADDPG is able to operate in environments that are either cooperative, competitive, or mixed, and even with agents with diverse capabilities. In this environment the agents interact cooperatively and they have identical capabilities, and so a single DDPG agent could have been trained to play both agents rather than MADDPG.

### Distributional Q-value estimation

In DQN we use the bellman equations to estimate the expected value of each state-action pair. Once these are known we can then simply take the action that yields the highest expected value, and this guarantees we perform optimally. In practice however these Q-values must be learned from data during training, and so convergence to correct Q-values can be improved by taking advantage of some of the generalisation properties of neural networks across different states. Although it's difficult to know eactly what a neural network is doing under the hood, it is likely that this effect is the cause of the great empirical performance of Distributional Q-value estimation. Here, instead of estimating the expected Q-value of a state-action pair we instead estimate the whole probability distribution of Q-values (https://arxiv.org/pdf/1804.08617.pdf). Thus the network is able to learn the overal shape of distributions typical to the specific environment during training, and then leverage this as a kind of baseline for training on other states with less data.

### n-step bootstrap targets
We also utilise a trick know to improve performance in environments where reward is delayed across many timesteps, such as this tennis environment. With n-step bootstrap targets instead of looking at the action->reward->next_state over a single timestep, we instead utilise a sequence of action->cumulated_n_rewards->nth_state. This essentially makes the agent learn from an action feedback cycle spanning several timesteps, so that it can experience future rewards immediately, rather than relying on Q-values to propogate back to the state one timestep ahead of it (https://arxiv.org/abs/1602.01783). Of course, a downside of this approach is that we must discard experience sequences of less than n steps, but in this environment there are typically many timesteps per episode and so this is not a problem. In this implementation n-step bootstrapping is implemented entirely within the ReplayBuffer class, which stores experience, cumulates rewards, and serves it to the agents for learning.

### asymmetric agent performance
It was observed during training that often one agent would 'figure something out', such as how to serve the ball, but the other agent wouldn't have yet learned it. This was bad because it would often result in the agents diverging while the better agent waits for the other to catch up and would usually forget what it learned. I added two small extensions to help prevent this problem:
1.  If an agents performance exceeded a certain threshold above the other agent, then it would stop updating it's networks until the other agent caught up. This prevents it forgetting in the meantime.
2.  If an agents performance exceeded a certain threshold above the other agent, then the worse agent would do a soft update of it's networks weights towards the better agent. This speeds up the catch up process and allows both agents to benefit from advancements made by one of them.
\
Since MADDPG is able to operate in environments where the two agents have asymmetric capabilities, for example a soccer shooter and a goalie, it's worth noting that the 1st extension would require tweaking of the reward thresholds, while the 2nd extension would not be possible.

### Final notes on features' effectiveness
Overall this approach has several dozen features over the base Q-learning algorithm, such as actor-critics, centralised multi-agent critic updates, n-step bootstrapping, alongside 20+ hyperparameters for things such as learning rates, pre-training, and the Replay buffer. Note however that the two most critical features for successful performance were found to be distributional Q-value estimation, and carefully chosen noise distribution and annealing.

### Glossary of papers
1.  DDPG algorithm https://arxiv.org/pdf/1509.02971.pdf
2.  Multi-Agent DDPG https://arxiv.org/pdf/1706.02275.pdf
3.  Distributional Q-learning https://arxiv.org/pdf/1804.08617.pdf
4.  n-step bootstrapping https://arxiv.org/pdf/1602.01783.pdf

## Further additions
Additions that might improve the algorithm further are those of the full MADDPG algorithm, such as:
1.  Prioritised Experience Replay https://arxiv.org/pdf/1511.05952.pdf
2.  Policy Ensembles https://arxiv.org/pdf/1706.02275.pdf
