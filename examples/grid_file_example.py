#!/usr/bin/env python

# Other imports.
import srl_example_setup
from simple_rl.tasks.grid_world import GridWorldMDPClass

mdp = GridWorldMDPClass.make_grid_world_from_file("pblocks_grid.txt", randomize=False)
mdp.visualize_value()
