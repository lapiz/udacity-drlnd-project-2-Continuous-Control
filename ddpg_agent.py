import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    initiailized = False
    memory = None
    actor_local = None
    actor_target = None
    actor_optimizer = None

    critic_local = None
    critic_target = None
    critic_optimizer = None

    def __init__(self, state_size, action_size, random_seed, hparams):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            params (dict) : hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.gamma = hparams['gamma']
        self.tau = hparams['tau']

        self.learn_per_step = hparams['learn_per_step']
        self.update_times = hparams['update_times']

        hidden_layers = hparams['hidden_layers']

        if Agent.initiailized is False:
            # Actor Network (w/ Target Network)
            Agent.actor_local = Actor(state_size, action_size, random_seed, hidden_layers['actor']).to(device)
            Agent.actor_target = Actor(state_size, action_size, random_seed, hidden_layers['actor']).to(device)
            Agent.actor_optimizer = optim.Adam(Agent.actor_local.parameters(), lr=hparams['lr']['actor'])

            # Critic Network (w/ Target Network)
            Agent.critic_local = Critic(state_size, action_size, random_seed, hparams['critic_activate'],hidden_layers['critic']).to(device)
            Agent.critic_target = Critic(state_size, action_size, random_seed, hparams['critic_activate'], hidden_layers['critic']).to(device)
            Agent.critic_optimizer = optim.Adam(Agent.critic_local.parameters(), lr=hparams['lr']['critic'], weight_decay=hparams['weight_decay'])


            # Replay memory
            Agent.memory = ReplayBuffer(action_size, hparams['buffer_size'], hparams['batch_size'], random_seed)

            Agent.initiailized = True


        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    
    def step( self, steps, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        Agent.memory.add(state, action, reward, next_state, done)

        if steps % self.learn_per_step != 0:
            return

        # Learn, if enough samples are available in memory
        if Agent.memory.is_enough_memory():
            for _ in range(self.update_times):
                experiences = Agent.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        Agent.actor_local.eval()
        with torch.no_grad():
            action = Agent.actor_local(state).cpu().data.numpy()
        Agent.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = Agent.actor_target(next_states)
        Q_targets_next = Agent.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = Agent.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        Agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        Agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = Agent.actor_local(states)
        actor_loss = -Agent.critic_local(states, actions_pred).mean()
        # Minimize the loss
        Agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        Agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(Agent.critic_local, Agent.critic_target)
        self.soft_update(Agent.actor_local, Agent.actor_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save_checkpoint(prefix):
        torch.save(Agent.actor_local.state_dict(), f'{prefix}_actor.pth')
        torch.save(Agent.critic_local.state_dict(), f'{prefix}_critic.pth')

    def load_checkpoint(prefix):
        Agent.actor_local.load_state_dict(torch.load(f'{prefix}_actor.pth'))
        Agent.critic_local.load_state_dict(torch.load(f'{prefix}_critic.pth'))

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def is_enough_memory(self):
        return len(self.memory) > self.batch_size
