#!/usr/bin/env python

# Python imports.
import sys

# Other imports
import srl_example_setup
from simple_rl.mdp import MDPDistribution
from simple_rl.tasks import GridWorldMDP, RandomMDP, ChainMDP, TaxiOOMDP, FourRoomMDP
from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from simple_rl.run_experiments import run_agents_lifelong
from simple_rl.utils import make_mdp

def main(open_plot=True):
    # Make MDP distribution, agents.
    mdp_distr = make_mdp.make_mdp_distr(mdp_class="four_room")
    ql_agent = QLearningAgent(actions=mdp_distr.get_actions())
    rand_agent = RandomAgent(actions=mdp_distr.get_actions())

    # Run experiment and make plot.
    run_agents_lifelong([ql_agent, rand_agent], mdp_distr, samples=10, episodes=1, steps=1000, reset_at_terminal=True, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
