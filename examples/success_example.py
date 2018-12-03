#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.planning import ValueIteration
from simple_rl.agents import QLearningAgent, RandomAgent, FixedPolicyAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):

    # Setup MDP.
    mdp = GridWorldMDP(width=8, height=3, init_loc=(1, 1), goal_locs=[(8, 3)], lava_locs=[(4, 2)], gamma=0.95, walls=[(2, 2)], slip_prob=0.05)

    # Make agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=20, episodes=300, steps=20, open_plot=open_plot, track_success=True, success_reward=1)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
