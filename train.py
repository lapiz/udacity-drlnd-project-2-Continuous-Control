import sys, platform, json
import numpy as np

from scores import Scores
from ddpg_agent import Agent
from unityagents import UnityEnvironment

def train(env, hparams ):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores_hparams = hparams['scores']
    scores = Scores( scores_hparams['expectation'],size=scores_hparams['window_size'], check_solved=scores_hparams['check_solved']) 

    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations                  # get the current state (for each agent)
    state_size = states.shape[1]

    agents = []
    for _ in range(num_agents):
        agents.append( Agent(state_size=state_size, action_size=action_size, random_seed=2, hparams=hparams))
    
    prefix = f'result/{hparams["output"]}'

    for i in range(hparams['epoch']):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        # number of agents
        num_agents = len(env_info.agents)
    
        # size of each action
        action_size = brain.vector_action_space_size
        states = env_info.vector_observations                  # get the current state (for each agent)

        # initialize the score (for each agent)
        epoch_score = np.zeros(num_agents)
        for t in range(hparams['t_max']):
            actions = np.array( [agents[i].act(states[i]) for i in range(num_agents) ])
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            dones = env_info.local_done                        # see if episode finished
            epoch_score += env_info.rewards                         # update the score (for each agent)
    
            for i in range(num_agents):
                agents[i].step(t, states[i], actions[i], env_info.rewards[i], next_states[i], dones[i]) 

            if np.any(dones):
                break
        if scores.AddScore(np.mean(epoch_score)) is True:
            break

    Agent.save_checkpoint(prefix)
    scores.FlushLog(prefix, False)


if __name__ == '__main__':
    config = 'default.json'
    if len(sys.argv) > 1:
        config = sys.argv[1]

    with open( config, 'r', encoding='utf-8') as f:
        hparams = json.load(f)

    print(hparams)
    fn = 'Reacher.app'
    if platform.system() == 'Linux':
        fn = '../Reacher_Linux_NoVis/Reacher.x86_64'
    env = UnityEnvironment(file_name=fn)    
    train(env, hparams)