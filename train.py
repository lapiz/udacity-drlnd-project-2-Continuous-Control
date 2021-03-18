import sys, platform
import numpy as np

from scores import Scores
from unityagents import UnityEnvironment


def train(env, epoch):
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
    
    train(env,epoch)