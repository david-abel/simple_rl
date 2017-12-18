#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearnerAgent, RandomAgent, DoubleQAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Setup MDP, Agents.
    mdp = GridWorldMDP(width=4, height=3, init_loc=(1,1), goal_locs=[(4,3)], gamma=0.95, walls=[(2,2)])

    dq_agent = DoubleQAgent(actions=mdp.get_actions())
    ql_agent = QLearnerAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, dq_agent, rand_agent], mdp, instances=25, episodes=100, steps=10, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not(sys.argv[-1] == "no_plot"))
