#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse
import matplotlib as plt
import numpy as np
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RMaxAgent, RandomAgent
from simple_rl.run_experiments import run_single_agent_on_mdp 
from simple_rl.tasks import FourRoomMDP, TaxiOOMDP, GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.planning import ValueIteration
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp
from collections import defaultdict

def main(open_plot=True):
    
    # Setup MDP.

    actual_args = {
        "width": 10, 
        "height": 10, 
        "init_loc": (1,1),
        "goal_locs": [(10,10)], 
        "lava_locs": [(1, 10), (3, 10), (5, 10), (7, 10), (9, 10)],
        "gamma": 0.9, 
        "walls": [(2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9),
                  (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9),
                  (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9),
                  (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9)],
        "slip_prob": 0.01,
        "lava_cost": 1.0,
        "step_cost": 0.1
    }

    mdp = GridWorldMDP(**actual_args)

    # Initialize the custom Q function for a q-learning agent. This should be equivalent to potential shaping.
    # This should cause the Q agent to learn more quickly.
    custom_q = defaultdict(lambda : defaultdict(lambda: 0))
    custom_q[GridWorldState(5,1)]['right'] = 1.0
    custom_q[GridWorldState(2,1)]['right'] = 1.0

    # Make a normal q-learning agent and another initialized with the custom_q above.
    # Finally, make a random agent to compare against.
    ql_agent = QLearningAgent(actions=mdp.get_actions(), epsilon=0.2, alpha=0.4)
    ql_agent_pot = QLearningAgent(actions=mdp.get_actions(), epsilon=0.2, alpha=0.4, custom_q_init=custom_q, name="PotQ")
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, ql_agent_pot, rand_agent], mdp, instances=2, episodes=60, steps=200, open_plot=open_plot, verbose=True)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")