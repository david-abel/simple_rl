#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RMaxAgent
from simple_rl.run_experiments import run_single_agent_on_mdp 
from simple_rl.tasks import FourRoomMDP, TaxiOOMDP, GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.planning import ValueIteration

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=str, default="learning", nargs='?', help="Choose the visualization type (one of {value, policy, agent, learning or interactive}).")
    args = parser.parse_args()
    return args.v

def main():
    
    # Setup MDP, Agents.
    mdp = GridWorldMDP(width=4, height=3, init_loc=(1, 1), goal_locs=[(4, 3)], lava_locs=[(4, 2)], gamma=0.95, walls=[(2, 2)], slip_prob=0.1)
    ql_agent = QLearningAgent(mdp.get_actions(), epsilon=0.2, alpha=0.2) 
    viz = parse_args()

    if viz == "value":
        # --> Color corresponds to higher value.
        # Run experiment and make plot.
        mdp.visualize_value()
    elif viz == "policy":
        # Viz policy
        value_iter = ValueIteration(mdp)
        value_iter.run_vi()
        policy = value_iter.policy
        mdp.visualize_policy(policy)
    elif viz == "agent":
        # --> Press <spacebar> to advance the agent.
        # First let the agent solve the problem and then visualize the agent's resulting policy.
        print("\n", str(ql_agent), "interacting with", str(mdp))
        run_single_agent_on_mdp(ql_agent, mdp, episodes=500, steps=200)
        mdp.visualize_agent(ql_agent)
    elif viz == "learning":
        # --> Press <r> to reset.
        # Show agent's interaction with the environment.
        mdp.visualize_learning(ql_agent, delay=0.005, num_ep=500, num_steps=200)
    elif viz == "interactive":
        # Press <1>, <2>, <3>, and so on to execute action 1, action 2, etc.
    	mdp.visualize_interaction()


if __name__ == "__main__":
    main()