# main code that contains the neural network setup and policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list, hard_update
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class MADDPG:
    def __init__(self, action_size, state_size, discount_factor=0.99, tau=0.0001, lr_actor=0.00025, lr_critic=0.0005, n_step = 5, num_atoms = 51, vmin = -0.1, vmax = 1, ):
        super(MADDPG, self).__init__()

        hidden_in_size, hidden_out_size = 300, 200
        
        self.maddpg_agent = [DDPGAgent(action_size, state_size, hidden_in_size, hidden_out_size, num_atoms, lr_actor, lr_critic), 
                             DDPGAgent(action_size, state_size, hidden_in_size, hidden_out_size, num_atoms, lr_actor, lr_critic)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.n_step = n_step
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = torch.linspace(self.vmin,self.vmax,self.num_atoms).to(device)
        self.atoms = self.atoms.unsqueeze(0)

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise_scale=np.zeros(2)):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise_scale[agent_num]) for agent_num, agent, obs in zip([0,1], self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update_targets(self,agent_num):
        """soft update targets"""
        self.iter += 1
        ddpg_agent = self.maddpg_agent[agent_num]
        soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
        soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
        
    def hard_update_targets(self,agent_num):
        """soft update targets"""
        self.iter += 1
        ddpg_agent = self.maddpg_agent[agent_num]
        hard_update(ddpg_agent.target_actor, ddpg_agent.actor)
        hard_update(ddpg_agent.target_critic, ddpg_agent.critic)
        
    def initialise_networks(self, agent0_num, agent0_path, agent1_num, agent1_path):
        
        checkpoints = [torch.load(agent0_path)[agent0_num],torch.load(agent1_path)[agent1_num]] # load the torch data
        
        for agent, checkpoint in zip(self.maddpg_agent,checkpoints):
        
            agent.actor.load_state_dict(checkpoint['actor_params'])                          # actor parameters
            agent.critic.load_state_dict(checkpoint['critic_params'])                        # critic parameters
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optim_params'])          # actor optimiser state
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optim_params'])        # critic optimiser state
            
            hard_update(agent.target_actor, agent.actor)                                     # hard updates to initialise the targets
            hard_update(agent.target_critic, agent.critic)                                   # hard updates to initialise the targets
            


    def update(self, samples, agent_number, logger):
        """
        Performs a distributional Actor/Critic calculation and update.
        Actor πθ and πθ'
        Critic Zw and Zw' (categorical distribution)
        """
        """update the critics and actors of all the agents """

        # need to transpose samples and convert them to trsnors
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples) # get data

        # full versions of obs and actions are needed for the critics
        obs_full = torch.cat(obs, 1)
        next_obs_full = torch.cat(next_obs, 1)
        action_full = torch.cat(action, 1)
        
        agent = self.maddpg_agent[agent_number] # set the agent being updated
        agent.critic_optimizer.zero_grad() # zero_grad is required before backprop

        target_actions = self.target_act(next_obs) # produces target actions using both critics for the next state, i.e. a(t+1)
        target_actions = torch.cat(target_actions, dim=1) # concatinates them together
        
        # Calculate target distribution to update the critic
        target_probs = agent.target_critic(next_obs_full, target_actions).detach() # target critic probs
        target_dist = self.to_categorical(reward[agent_number].unsqueeze(-1), target_probs, done[agent_number].unsqueeze(-1))
        
        # Calculate log probability distribution
        log_probs = agent.critic(obs_full, action_full, log=True) # for cross entropy we need log of critic output
        
        # Calculate the critic loss manually using cross entropy
        critic_loss = -(target_dist * log_probs).sum(-1).mean()
        
        # Update critic using gradient descent
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1) # clipping to prevent too big updates
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # get actions of all agents
        # detach the other agents to save computation when computing derivative
        actor_actions = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ] 
                
        # combine all the actions and observations for input to critic
        actor_actions = torch.cat(actor_actions, dim=1)

        critic_probs = agent.critic(obs_full, actor_actions)
        
        # Multiply value probs by atom values and sum across columns to get Q-Value
        expected_reward = (critic_probs * self.atoms).sum(-1)
        # Calculate the actor network loss
        actor_loss = -expected_reward.mean()
        actor_loss.backward(retain_graph=True)
        agent.actor_optimizer.step()
        
        # adding the losses to tensorboard
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,{'critic loss': cl,'actor_loss': al},self.iter)
        
        
    def to_categorical(self, rewards, probs, dones):
        """
        Credit to Matt Doll and Shangtong Zhang for this function:
        https://github.com/whiterabbitobj
        https://github.com/ShangtongZhang
        """

        # Create local vars to keep code more concise
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms
        n_step = self.n_step
        discount_factor = self.discount_factor
        
        # this is the increment between atoms
        delta_z = (vmax - vmin) / (num_atoms - 1)

        # projecting the rewards to the atoms
        projected_atoms = rewards + discount_factor**n_step * atoms * (1 - dones)
        projected_atoms.clamp_(vmin, vmax) # vmin/vmax are arbitary so any observations falling outside this range will be cliped
        b = (projected_atoms - vmin) / delta_z

        # precision is reduced to prevent e.g 99.00000...ceil() evaluating to 100 instead of 99.
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        # initialising projected_probs
        projected_probs = torch.tensor(np.zeros(probs.size())).to(device)

        # a bit like one-hot encoding but for the specified atoms
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()

    
    
    # old update for non-distributional ddpg which was originally used. Can be safely deleted or ignored
    def old_ddpg_update(self, samples, agent_number, logger): 
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples) # get data

        # full versions of obs are needed for the critics
        obs_full = torch.cat(obs, 1)
        next_obs_full = torch.cat(next_obs, 1)
        
        agent = self.maddpg_agent[agent_number] # set the agent
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(s(t+1),a(t+1)) from target network
        # where a(t+1) is from actor networks
        
        target_actions = self.target_act(next_obs) # produces target actions using both critics for the next state, i.e. a(t+1)
        target_actions = torch.cat(target_actions, dim=1) # concatinates them together
        
        #target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full.t(),target_actions) # the Q value for next state, i.e. Q(s(t+1),a(t+1))
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1)) # y = Rt + gamma*Q(t+1)
        action = torch.cat(action, dim=1) # the action actually taken
        #critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(obs_full, action) # 

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach()) # calculating the loss between Q(st,at) and y
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # get actions of all agents
        # detach the other agents to save computation when computing derivative
        actor_actions = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ] 
                
        actor_actions = torch.cat(actor_actions, dim=1) # concat actions as input to critic
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        #q_input2 = torch.cat((obs_full.t(), actor_actions), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(obs_full, actor_actions).mean() # the actor is updated using gradient ascent of the critic
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)