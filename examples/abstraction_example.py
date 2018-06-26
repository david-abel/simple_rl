#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.abstraction import AbstractionWrapper

def main(open_plot=True):
    # Setup MDP, Agents.
    mdp = GridWorldMDP(width=10, height=10, init_loc=(1, 1), goal_locs=[(10, 10)])
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())
    abstr_identity_agent = AbstractionWrapper(QLearningAgent, agent_params={"epsilon":0.9, "actions":mdp.get_actions()})

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent, abstr_identity_agent], mdp, instances=5, episodes=100, steps=150, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
