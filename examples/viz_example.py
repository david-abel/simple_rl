#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearnerAgent
from simple_rl.run_experiments import run_single_agent_on_mdp 
from simple_rl.tasks import FourRoomMDP, GridWorldMDP
from simple_rl.planning import ValueIteration
from simple_rl.utils.make_mdp import make_mdp_distr

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type = str, default="value", nargs = '?', help = "Choose the visualization type (one of {value, policy, agent}).")
    args = parser.parse_args()
    return args.v

def main():
    # Setup MDP, Agents.
    mdp = FourRoomMDP(6, 6, goal_locs=[(6, 6)], gamma=0.9)
    ql_agent = QLearnerAgent(mdp.get_actions())
    viz = parse_args()

    if viz == "value":
        # Run experiment and make plot.
        mdp.visualize_value()
    elif viz == "policy":
        # Viz policy
        vi = ValueIteration(mdp)
        vi.run_vi()
        policy = vi.policy
        mdp.visualize_policy(policy)
    elif viz == "agent":
        # Solve problem and show agent interaction.
        print("\n", str(ql_agent), "interacting with", str(mdp))
        run_single_agent_on_mdp(ql_agent, mdp, episodes=500, steps=200, open_plot=False)
        mdp.visualize_agent(ql_agent)

if __name__ == "__main__":
    main()
