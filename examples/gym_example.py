#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
import srl_example_setup
from simple_rl.agents import RandomAgent, LinearQAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Gym MDP
    gym_mdp = GymMDP(env_name='Breakout-v0', render=False)
    num_feats = gym_mdp.get_num_state_feats()

    # Setup agents and run.
    rand_agent = RandomAgent(gym_mdp.get_actions())
    dqn_agent = LinearQAgent(gym_mdp.get_actions(), num_feats)
    run_agents_on_mdp([dqn_agent, rand_agent], gym_mdp, instances=3, episodes=20, steps=200, open_plot=open_plot, verbose=True)

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
