#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RMaxAgent
from simple_rl.run_experiments import run_single_agent_on_mdp 
from simple_rl.tasks import FourRoomMDP, TaxiOOMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.planning import ValueIteration

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=str, default="value", nargs='?', help="Choose the visualization type (one of {value, policy, agent}).")
    args = parser.parse_args()
    return args.v

def main():
    
    # Setup MDP, Agents.
    mdp = FourRoomMDP(11, 11, goal_locs=[(11, 11)], gamma=0.9, step_cost=0.0)
    ql_agent = QLearningAgent(mdp.get_actions(), epsilon=0.2, alpha=0.4) 
    viz = parse_args()

    # Choose viz type.
    viz = "learning"

    if viz == "value":
        # Run experiment and make plot.
        mdp.visualize_value()
    elif viz == "policy":
        # Viz policy
        value_iter = ValueIteration(mdp)
        value_iter.run_vi()
        policy = value_iter.policy
        mdp.visualize_policy(policy)
    elif viz == "agent":
        # Solve problem and show agent interaction.
        print("\n", str(ql_agent), "interacting with", str(mdp))
        run_single_agent_on_mdp(ql_agent, mdp, episodes=500, steps=200)
        mdp.visualize_agent(ql_agent)
    elif viz == "learning":
        # Run experiment and make plot.
        mdp.visualize_learning(ql_agent)
    elif viz == "interactive":
    	mdp.visualize_interaction()


if __name__ == "__main__":
    main()
