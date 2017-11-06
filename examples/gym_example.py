#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
import srl_example_setup
from simple_rl.agents import LinearQLearnerAgent, RandomAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Gym MDP
    gym_mdp = GymMDP(env_name='CartPole-v0', render=False)
    num_feats = gym_mdp.get_num_state_feats()

    # Setup agents and run.
    lin_agent = LinearQLearnerAgent(gym_mdp.actions, num_features=num_feats, alpha=0.4, epsilon=0.4, anneal=True)
    rand_agent = RandomAgent(gym_mdp.actions)
    run_agents_on_mdp([lin_agent, rand_agent], gym_mdp, instances=10, episodes=30, steps=10000, open_plot=open_plot)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    # logging.basicConfig(level=logging.WARNING)
    main(open_plot=not(sys.argv[-1] == "no_plot"))
