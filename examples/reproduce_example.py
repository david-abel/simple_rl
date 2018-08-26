#!/usr/bin/env python

# Python imports.
import sys, os

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp, reproduce_from_exp_file

def main(open_plot=True):

    # Reproduce the experiment.
    reproduce_from_exp_file(exp_name="gridworld_h-3_w-4", open_plot=open_plot, results_dir_name=os.path.join(os.path.dirname(os.path.realpath(__file__)), "experiments_to_reproduce"))

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
