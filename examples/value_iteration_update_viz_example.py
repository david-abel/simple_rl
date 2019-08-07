#!/usr/bin/env python

# Python imports
import matplotlib as plt
import numpy as np
import time
import argparse

# Other imports
import srl_example_setup
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning.ValueIterationClass import ValueIteration

# INSTRUCTIONS FOR USE:
# 1. When you run the program [either with default or supplied arguments], a pygame window should pop up.
#    This is the first iteration of running VI on the given MDP.
# 2. Press any button to close this pygame window and wait, another window will pop-up displaying the 
#    policy from the next time step
# 3. Repeat 1 and 2 until the program terminates

# An input function, creates the MDP object based on user input
def generate_MDP(width, height, init_loc, goal_locs, lava_locs, gamma, walls, slip_prob):
    """ Creates an MDP object based on user input """

    actual_args = {
        "width": width, 
        "height": height, 
        "init_loc": init_loc,
        "goal_locs": goal_locs, 
        "lava_locs": lava_locs,
        "gamma": gamma, 
        "walls": walls, 
        "slip_prob": slip_prob,
        "lava_cost": 1.0,
        "step_cost": 0.1
    }

    return GridWorldMDP(**actual_args)

def main():
    # This accepts arguments from the command line with flags.
    # Example usage: python value_iteration_demo.py -w 4 -H 3 -s 0.05 -g 0.95 -il [(0,0)] -gl [(4,3)] -ll [(4,2)]  -W [(2,2)]
    parser = argparse.ArgumentParser(description='Run a demo that shows a visualization of value iteration on a GridWorld MDP')

    # Add the relevant arguments to the argparser
    parser.add_argument('-w', '--width', type=int, nargs="?", const=4, default=4,
    help='an integer representing the number of cells for the GridWorld width')
    parser.add_argument('-H', '--height', type=int, nargs="?", const=3, default=3,
    help='an integer representing the number of cells for the GridWorld height')
    parser.add_argument('-s', '--slip', type=float, nargs="?", const=0.05, default=0.05,
    help='a float representing the probability that the agent will "slip" and not take the intended action')
    parser.add_argument('-g', '--gamma', type=float, nargs="?", const=0.95, default=0.95,
    help='a float representing the decay probability for Value Iteration')
    parser.add_argument('-il', '--i_loc', type=tuple, nargs="?", const=(0,0), default=(0,0),
    help='two integers representing the starting cell location of the agent [with zero-indexing]')
    parser.add_argument('-gl', '--g_loc', type=list, nargs="?", const=[(3,3)], default=[(3,3)],
    help='a sequence of integer-valued coordinates where the agent will receive a large reward and enter a terminal state')
    parser.add_argument('-ll', '--l_loc', type=list, nargs="?", const=[(3,2)], default=[(3,2)],
    help='a sequence of integer-valued coordinates where the agent will receive a large negative reward and enter a terminal state')
    parser.add_argument('-W', '--Walls', type=list, nargs="?", const=[(2,2)], default=[(2,2)],
    help='a sequence of integer-valued coordinates representing cells that the agent cannot transition into')

    args = parser.parse_args()
    mdp = generate_MDP(
        args.width, 
        args.height,
        args.i_loc,
        args.g_loc,
        args.l_loc,
        args.gamma, 
        args.Walls, 
        args.slip)

    # Run value iteration on the mdp and save the history of value backups until convergence
    vi = ValueIteration(mdp, max_iterations=50)
    _, _, histories = vi.run_vi_histories()

    # For every value backup, visualize the policy
    for value_dict in histories:
        mdp.visualize_policy(lambda in_state: value_dict[in_state]) # Note: This lambda is necessary because the policy must be a function
        time.sleep(0.5)

if __name__ == "__main__":
    main()