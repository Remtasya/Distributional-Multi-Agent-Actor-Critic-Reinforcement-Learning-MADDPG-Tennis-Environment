# Report

This report is designed to cover the following:
1.  Theoretical MADDPG Agent Design
2.  Implementation, Hyperparameters, and Performance

## Theoretical MADDPG Agent Design

The algorithm used is based on the Deep Reinforcement Learning MADDPG, DDPG, D4PG algorithms described in these papers:
1.  MADDPG - https://arxiv.org/pdf/1706.02275.pdf
2.  DDPG - https://arxiv.org/pdf/1509.02971.pdf
3.  D4PG - https://arxiv.org/pdf/1804.08617.pdf
\

Details of these algorithms are below:

### Actor-Critic methods
Deep Q-Learning is designed for environments with discreet action spaces, but struggles to generalise to continuous action spaces. This is because Q-Learning works by computing all possible state-actions values and then choosing the highest one. In continuous action spaces Q-value iteration is not possible and instead would require a continuous optimisation to select best actions. Actor-Critic methods rememdy this problem by using two networks instead of just the Q-network:

The **Critic Network** is a modified Q-network estimator that is designed to output the value for any given state-action value rather than iterating through all of them. As in Q-Learning, it uses the Bellman equations for learning the Q-values.

The **Actor network** attempts to estimate the optimal action directly given a state, and is used for selecting which actions to take. It is learned by making use of the Critic Network as a baseline, where the critic evaulates how good the actor's action is.

### Off-policy learning
In addition to using an actor-critic setup, the Deep Deterministic Policy Gradient algorithm (https://arxiv.org/pdf/1509.02971.pdf) additionally makes use of the success of Deep Q-networks to incorporate off-policy learning by use of a replay buffer and target networks for both the Critic and Actor which are updated periodically. These modification enable more stable learning, and are described below:

#### Replay Buffer
The replay buffer functions as a large collection of experiences observed by the agent, updated after each action. Training of the networks then proceeds by uniformly sampling minibatches from the replay buffer, and performing small gradient updates from these. This approach is prefered to online learning because it attempts to de-correlate the experiences, and so doesn't violate the neural network update procedure assumption of independently selected samples in each minibatch.

#### N-step Bootstrap Targets
In the replay buffer we also utilise a trick know to improve performance in environments where reward is delayed across many timesteps, such as this tennis environment. With n-step bootstrap targets instead of looking at the action->reward->next_state over a single timestep, we instead consider a gap of n steps, i.e. action->cumulated_n_rewards->nth_state. This essentially makes the agent learn from an action feedback cycle spanning several timesteps, so that it can experience future rewards immediately, rather than relying on Q-values to propogate back to the state one timestep ahead of it (https://arxiv.org/abs/1602.01783). Of course, a downside of this approach is that we must discard experience sequences of less than n steps, but in this environment there are typically many timesteps per episode and so this is not a problem. In this implementation n-step bootstrapping is implemented entirely within the ReplayBuffer class, which stores experience, cumulates rewards, and serves it to the agents for learning.

#### Target Networks
Off-policy target networks are also used to improve stability of the learning algorithm. This works by using target networks in the Q-targets formula for estimating the value of future states, which is used to train the current networks. The target networks are only updated slowly to the current networks, which helps prevent optimistic bias when estimating Q-values, which can lead to divergence.

### Multi-Agent Deep Deterministic Policy Gradients

A multi-agent setting introduces several additional complexities. For example, from the perspective of a single agent, it is tempting to view other agents as part of the environment,as if they merely receive an action and output a state response. Note however that because the other agents are updating over the course of training, their behaviour is changing. Thus off policy training - such as experience replay, which is crucial for stablising DDPG - would violate a key assumption of Reinforcement Learning of having a stationary environment. The MADDPG paper (https://arxiv.org/pdf/1706.02275.pdf) introduces the concept of centralised training but de-centralised execution as a work-around for this problem. Essentially each critic is augmented to receive as inputs the actions and states of all other agents as well as it's own, and during updates utilises other agent's actors and target actors rather than viewing them as part of the environment. This omniscient critic is used to train the actor, by performing gradient ascent on the critic's evaluation of the actor's action. During execution outside of training though the actor is able to act freely without access to other agent's internal models.

\
Note that MADDPG also includes some other additions, such as prioritised experience replay and policy ensembles, which were not used in this implementation. Also noteworthy is the fact that MADDPG is able to operate in environments that are either cooperative, competitive, or mixed, and even with agents with diverse capabilities. In this environment the agents interact cooperatively and they have identical capabilities, and so a single DDPG agent could have been trained to play both agents rather than MADDPG.

### Distributional Q-value Estimation

In DQN we use the bellman equations to estimate the expected value of each state-action pair. Once these are known we can then simply take the action that yields the highest expected value, and this guarantees we perform optimally. In practice however these Q-values must be learned from data during training, and so convergence to correct Q-values can be improved by taking advantage of some of the generalisation properties of neural networks across different states. Although it's difficult to know eactly what a neural network is doing under the hood, it is likely that this effect is the cause of the great empirical performance of Distributional Q-value estimation. Here, instead of estimating the expected Q-value of a state-action pair we instead estimate the whole probability distribution of Q-values (https://arxiv.org/pdf/1804.08617.pdf). Thus the network is able to learn the overal shape of distributions typical to the specific environment during training, and then leverage this as a kind of baseline for training on other states with less data.

### Noise

Noise is critical to good performance of an RL agent as it determines how the trade-off between exploration and exploitation is managed. Too little noise can result in the agent getting stuck in local optima indefinitely during training, while too much noise can result in catastrophic forgetting of good learned behaviour. This script contains 4 types of noise to choose from: 
1.  OUNoise - which produces serially correlated noise with a mean-reverting property. Because it is serially correlated it can produce actions that are smoother in appearance than other types of noise - for example uniform noise will appear jittery by comparison. An example of the serial correlation of the OB process can be seen below: <img src="https://github.com/Remtasya/MADDPG/blob/master/project_images/OU process.PNG" width="600"/>
2.  GaussNoise - which simply adds standard white noise which is then clipped to the -1 to 1 interval.
3.  WeightedNoise - which simply takes a weighted average of uniform noise on the range -1 to 1 with the action.
All of these noises differ in the extent to which they anneal their variance, and favour the center i.e. 0 of the interval versus respect the action value with their mean. They can easily be experimented with to determine the most suitable choice for an environment.
4.  BetaNoise - which transforms the network action using a beta distribution. Because the beta distribution has a support from 0 to 1 it can easily be extended to -1 to 1, which makes it able to provide variance and mean adaptive noise in both directions without requiring clipping to the -1 to 1 interval. The beta distribution's abilty to adapt it's mean within the 0 to 1 support can be seen below: 
<img src="https://github.com/Remtasya/MADDPG/blob/master/project_images/beta.gif" width="400"/>

As with other techniques such as epsilon-greedy exploration in DQN, it can be useful to anneal the noise over the course of training. This encourages random exploration of the state space early on and better convergence later on. In this implementation, each type of noise is decayed after each episode, and this was found to be essential to solving the environment.


### Asymmetric Agent Performance
It was observed during training that often one agent would 'figure something out', such as how to serve the ball, but the other agent wouldn't have yet learned it. This was bad because it would often result in the agents diverging while the better agent waits for the other to catch up and would usually forget what it learned. I added two small extensions to help prevent this problem:
1.  If an agents performance exceeded a certain threshold above the other agent, then it would stop updating it's networks until the other agent caught up. This prevents it forgetting in the meantime.
2.  If an agents performance exceeded a certain threshold above the other agent, then the worse agent would do a soft update of it's networks weights towards the better agent. This speeds up the catch up process and allows both agents to benefit from advancements made by one of them.
\
Since MADDPG is able to operate in environments where the two agents have asymmetric capabilities, for example a soccer shooter and a goalie, it's worth noting that the 1st extension would require tweaking of the reward thresholds, while the 2nd extension would not be possible.


### Final Notes on Features' Effectiveness
Overall this approach has several dozen features over the base Q-learning algorithm, such as actor-critics, centralised multi-agent critic updates, n-step bootstrapping, alongside 20+ hyperparameters for things such as learning rates, pre-training, and the Replay buffer. Note however that the two most critical features for successful performance were found to be distributional Q-value estimation, and carefully chosen noise distribution and annealing.



## Implementation and Empirical Results

After ~2600 episodes the agent was about to 'solve' the enviroment by attaining an average reward over 100 episodes greater than 0.5.

A plot of score over time is shown below:

<img src="https://github.com/Remtasya/MADDPG/blob/master/project_images/performance.PNG" width="400"/>

It can be seen that it is able to achieve 1.5+ average score arund the 3200 episode mark. In fact in the testing environment with no noise it can achieve up to 2.6+ average score, and so long as it managed to serve the ball OK it can keep rallying indefinitely.

Note also that performance is very temperamental and it appears to suffer from catastrophic forgetting after 4000 episodes.

### Neural Network Architecture

The architectures used for both the Actor and Critic are simple feed-forward Neural Networks.

**For the Actor**

1.  Input state data which has 24 dimensions
2.  The 1st hidden Linear layer with 300 neurons followed by relu activation
3.  The 2nd hidden Linear layer with 200 neurons followed by relu activation
4.  The output layer with 2 neurons corresponding to the action dimensions, with tanh activation to ensure output in the (-1,1) range.

**For the Critic**
1.  Input state data which has 24x2 dimensions
2.  The 1st hidden Linear layer with 300 neurons followed by relu activation
3  The previous hidden layer output is concatinated with the action which has dimension 2x2, which is used as input for this layer
4.  The 2nd hidden Linear layer with 200 neurons followed by relu activation
5.  The output layer with 51 neurons corresponding to the probability distribution of the state-action value, with softmax activation to ensure it is a valid probability distribution (i.e. probs in the range 0 to 1 and they all sum to 1).


## Hyperparameters
#### Several Hyperparameters were used in this implementation which will be described below:

**Episode parameters:**

**number_of_episodes  (int): maximum number of training episodes**\
1000-10000 episodes were typically required to see good performance

**episode_length  (int): maximum number of timesteps per episode**\
 1000 timesteps is a reasonable number
 
 **episodes_before_training   (int): number of purely-random episodes before training**\
 1000 episodes of purely random behaviour was chosen. This was found to improve initial learning as it fills the replay buffer with exploration.

**Replay Buffer parameters:**

 **buffer size   (int): the maximum size of the buffer, after which samples are discarded**\
 50000 represents a few thousand episodes of training early on.

 
 **n_steps   (int): the number n of timesteps used for the bootstrap targets**\
5 was chosen after experiments.


**discount_rate (float): discount rate**\
Close to 1 will cause the agent to value all future rewards equally, while close to 0 will cause the agent to prioritise more immediate rewards. Unlike most hyperparameters, this will not only effect convergence but also the optimal policy converged to. Close to 1 is often best, so 0.99 is chosen.

**Model hyperparameters:**

**lr_actor (float): model hyperparameter - learning rate**\
This determines how large the model weight updates of the actor network are after each learning step. Too large and instability is caused, while too small and the model may never converge. I chose 0.00025

**lr_critic (float): model hyperparameter - learning rate**\
The same as above for the critic, except a slightly higher learning rate of 0.0005 is chosen.

**batchsize (int): model hyperparameter - number of experiences sampled for a model minibatch**\
Too low will cause learning instability and poor convergence, too high can cause convergence to local optima. I chose 256 as a default.

**tau (float): model hyperparameter - how closely the target-network should track the current network**\
After every learning step the target-network weights are updated closer to the current network, so that the target-network weights are a moving average over time of the current network past weights. i chose a relatively small value, 0.0001, although haven't experimented with tuning it.

**l2_decay  (float): model hyperparameter - decay for the adaptive learning rates for the networks**\
0.0001 slightly improved performance

**hidden_in_size  (int): model hyperparameter - size of the first hidden layer**\
300 nodes

**hidden_out_size  (int): model hyperparameter - size of the first hidden layer**\
200 nodes

**catchup_tau  (int): model hyperparameter - determines catch-up rate of worse agent**\
Every timestep, if there is a significant disparity between the agents performance then the worse one will update towards the better one with weight 0.01.

**num_atoms (int): model hyperparameter - number of bins in the Q-function approximation**\
51 is the number in the original c51 algorithm.

**vmin & vmax (float): model hyperparameter - number of bins in the Q-function approximation**\
These are the min and max values possible for Q-values, which represented the discounted present value of all future rewards. Since the agent gets -0.01 for dropping the ball this is the lowest possible value, while the highest value could be something like 1-5 depending on the timesteps per episode.

**noise parameters:**

**noise_reduction  (float): how much to decay noise each episode**\
By using a decay of 0.998 noise will gradually decay during training. The goal is to balance adequate early exporation with precise later convergence. Note that each noise type utilises nose_scale in a different manner, but all have a similar effect.

**noise_scale_end  (float): noise strength added at each timestep to the actions**\
As is common with exploration, we don't want our agent to ever stop exploring all together, and so noise level will never deop below 0.001.

**noise_type  (float): the type of noise utilised**\
'BetaNoise' is the default as I found it to be most effective, but the others work as well.

**OU_mu (float): long term mean of the OU process**\
This determines the long-term mean that the OU process will revert to. Default 0 since this represents no noise added to the actions

**OU_theta (float): mean reversion strength of the OU process**\
The strength of the mean reversion. Default from the OU paper is 0.15

**OU_sigma (float): noise strength added at each timestep to the OU process**\
The strength of the noise - default is 0.2.



