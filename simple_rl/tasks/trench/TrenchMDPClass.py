''' TrenchMDPClass.py: Contains the Trench class. '''

# Python imports.
import random
import math

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class TrenchMDP(GridWorldMDP):
    ''' Class for a Trench '''

    def __init__(self, width=9, height=9, init_loc=(1,1), goal_locs=[(9,9)], gamma=0.99, slip_prob=0.00, name="trench"):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        GridWorldMDP.__init__(self, width, height, init_loc, goal_locs, walls=self._compute_walls(width, height), gamma=gamma, slip_prob=slip_prob, name=name)

    