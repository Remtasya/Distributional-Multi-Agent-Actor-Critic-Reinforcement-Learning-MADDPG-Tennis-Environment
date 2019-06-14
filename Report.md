# Report

This report is designed to cover the following in more detail than the readme:
1.  Theoretical DDPG Agent Design
2.  Implementation, Hyperparameters, and Performance
3.  Ideas for Future Improvements

## Theoretical DDPG Agent Design

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

### Off-policy learning
In addition to using an actor-critic setup, the Deep Deterministic Policy Gradient algorithm (https://arxiv.org/pdf/1509.02971.pdf) additionally makes use of the success of Deep Q-networks to incorporate off-policy learning by use of a replay buffer and target networks for both the Critic and Actor which are updated periodically. These modification enable more stable learning, and are described below:

#### Replay Buffer
The replay buffer functions as a large collection of experiences observed by the agent, updated after each action. Training of the networks then proceeds by uniformly sampling minibatches from the replay buffer, and performing small grandient updates from these. This approach is prefered to online learning because it attempts to de-correlate the experiences, and so doesn't violate the neural network update procedure assumption of independently selected samples in each minibatch.

#### Target Networks
Off-policy target networks are also used to improve stability of the learning algorithm. This works by using target networks in the Q-targets formula for estimating the value of future states, which is used to train the current networks. The target networks are only updated slowly to the current networks, which helps prevent optimistic bias when estimating Q-values, which can lead to divergence.


### Ornstein-Uhlenbeck process

To encourage early exploration an Ornstein-Uhlenbeck (OU) process is used to add noise to the actions, which is then clipped to be in the (-1,1) range. An OU process is a type of auto-correlated noise process, where noise from past observations is compounded with new random noise, with a mean-reverting trajectory.

Below a plot shows a typical trajectory of an OU process compared to a guassian process - the OU is auto-correlated whereas the guassian is serially indepedent.

<img src="https://github.com/Remtasya/DDPG-Actor-Critic-Reinforcement-Learning-Reacher-Environment/blob/master/project_images/OU process.PNG" alt="Ornstein-Uhlenbeck" width="700"/>

As with other techniques such as epsilon-greedy exploration in DQN, it can be useful to anneal the noise over the course of training. This encourages random exploration of the state space early on and better convergence later on. In this implementation, the std. dev. of the noise is decayed after each episode, and this was found to be essential to solving the environment.


## Implementation and Empirical Results

After ~110 episodes the agent was about to 'solve' the enviroment by attaining an average reward over 100 episodes greater than 30.0.

A plot of score over time is shown below:

<img src="https://github.com/Remtasya/DDPG-Actor-Critic-Reinforcement-Learning-Reacher-Environment/blob/master/project_images/performance.PNG" alt="Environment" width="400"/>


### Neural Network Architecture

The architectures used for both the Actor and Critic are simple feed-forward Neural Networks.

**For the Actor**

1.  Input state data which has 33 dimensions
2.  The 1st hidden Linear layer with 400 neurons followed by relu activation
3.  The 2nd hidden Linear layer with 300 neurons followed by relu activation
4.  The output layer with 4 neurons corresponding to the action dimensions, with tanh activation to ensure output in the (-1,1) range.

**For the Critic**
1.  Input state data which has 33 dimensions
2.  The 1st hidden Linear layer with 400 neurons followed by relu activation
3  The previous hidden layer output is concatinated with the action which has dimension 4, which is used as input for this layer
4.  The 2nd hidden Linear layer with 300 neurons followed by relu activation
5.  The output layer with 1 neuron corresponding to the value of the state-action value, with no activation


## Hyperparameters
#### Several Hyperparameters were used in this implementation which will be described below:




**n_episodes (int): maximum number of training episodes**\
the model was found to converge after ~110 episodes.

**max_t (int): maximum number of timesteps per episode**\
this environment terminates ~1000 timseteps

**GAMMA (float): discount rate**\
Close to 1 will cause the agent to value all future rewards equally, while close to 0 will cause the agent to prioritise more immediate rewards. Unlike most hyperparameters, this will not only effect convergence but also the optimal policy converged to. Close to 1 is often best, so 0.99 is chosen.

**LR_ACTOR (float): model hyperparameter - learning rate**\
This determines how large the model weight updates of the actor network are after each learning step. Too large and instability is caused, while too small and the model may never converge. I chose 5e-4.

**LR_CRITIC (float): model hyperparameter - learning rate**\
The same as above for the critic, except a slightly higher learning rate of 1e-3 is chosen.

**BATCH_SIZE (int): model hyperparameter - number of experiences sampled for a model minibatch**\
Too low will cause learning instability and poor convergence, too high can cause convergence to local optima. I chose 128 as a default.

**BUFFER_SIZE (int): replay buffer size**\
this is the size of the experience buffer, which when exceeded will drop old experiences. This is mainly limited by your available RAM - if you experience issues with RAM try lowering it

**TAU (float): how closely the target-network should track the current network**\
After every learning step the target-network weights are updated closer to the current network, so that the target-network weights are a moving average over time of the current network past weights. i chose a relatively small value (1e-3) although haven't experimented with tuning it.


**Ornstein-Uhlenbeck process noise parameters:**

**mu (float): long term mean of the OU process**\
This determines the long-term mean that the OU process will revert to. Default 0 since this represents no noise added to the actions

**theta (float): mean reversion strength of the OU process**\
The strength of the mean reversion. Default from the OU paper is 0.15

**sigma (float): noise strength added at each timestep to the OU process**\
The strength of the noise - default is 0.2.

**SIGMA_DECAY (float): how much to dcay sigma each episode**\
By using a decay of 0.95 sigma will gradually decay during training. The goal is to balance adequate early exporation with precise later convergence.

**SIGMA_MIN (float): noise strength added at each timestep to the OU process**\
As is common with exploration, we don't want our agent to ever stop exploring all together, and so sigma will never deop below 0.005.
## Ideas for Future Improvements

Additions that might improve the algorithm further are those of the D4PG algorithm, which inculdes prioritised experience replay, distributional value-learning, and n-step bootstrapping.

**You can read more about these here:**

1.  Prioritised Experience Replay - https://arxiv.org/abs/1511.05952
2.  Distributional DQN - https://arxiv.org/abs/1707.06887
3.  Learning from multi-step bootstrap targets -  https://arxiv.org/abs/1602.01783

