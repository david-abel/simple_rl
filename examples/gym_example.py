#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
import srl_example_setup
from simple_rl.agents import LinearQLearningAgent, RandomAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Gym MDP
    gym_mdp = GymMDP(env_name='Acrobot-v1', render=True)
    num_feats = gym_mdp.get_num_state_feats()

    # Setup agents and run.
    lin_agent = LinearQLearningAgent(gym_mdp.get_actions(), num_features=num_feats, rbf=True, alpha=0.1, epsilon=0.1, anneal=True)
    rand_agent = RandomAgent(gym_mdp.get_actions())
    run_agents_on_mdp([lin_agent, rand_agent], gym_mdp, instances=50, episodes=50, steps=100, open_plot=open_plot)

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
