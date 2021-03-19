import sys, platform
import numpy as np

from scores import Scores
from ddpg_agent import Agent
from unityagents import UnityEnvironment


default_hparams = {
    'buffer_size':int(1e5),  # replay buffer size
    'batch_size':128,        # minibatch size
    'gamma':0.99,            # discount factor
    'tau':1e-3,              # for soft update of target parameters'
    'lr': { 
        'actor':1e-4,        # learning rate of the actor 
        'critic':1e-3,       # learning rate of the critic
    },
    'weight_decay':0,        # L2 weight decay

    'hidden_layers': {
        'actor': [ 400,300 ],
        'critic': [ 400,300 ],
    },
    'critic_activate': 'relu'
}


def train(env, epoch, hparams):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores = Scores(10,size=100, check_solved=False) 

    for i in range(epoch):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        # number of agents
        num_agents = len(env_info.agents)
    
        # size of each action
        action_size = brain.vector_action_space_size
        states = env_info.vector_observations                  # get the current state (for each agent)

        # initialize the score (for each agent)
        epoch_score = np.zeros(num_agents)
        done = False
        while not done:
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            dones = env_info.local_done                        # see if episode finished
            epoch_score += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            done = np.any(dones)                              # exit loop if episode finished
        scores.AddScore(np.mean(epoch_score))



if __name__ == '__main__':
    fn = 'Reacher.app'
    if platform.system() == 'Linux':
        fn = 'Reacher_Linux_NoVis/Reacher.x86_64'
    env = UnityEnvironment(file_name=fn)    
    epoch = 1000
    if len(sys.argv) > 1:
        epoch = int(sys.argv[1])
    
    train(env,epoch, default_hparams)