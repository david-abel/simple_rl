#!/usr/bin/env python

# Python imports.
import sys, os

# Other imports.
import srl_example_setup
from simple_rl.run_experiments import reproduce_from_exp_file

def main(open_plot=True):

	# Grab results directory.
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "experiments_to_reproduce")

    # Reproduce the experiment.
    reproduce_from_exp_file(exp_name="gridworld_h-3_w-4", results_dir=results_dir, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
