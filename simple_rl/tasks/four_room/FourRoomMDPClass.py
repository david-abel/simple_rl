''' FourRoomMDPClass.py: Contains the FourRoom class. '''

# Python imports.
import math

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class FourRoomMDP(GridWorldMDP):
    ''' Class for a FourRoom '''

    def __init__(self, width=9, height=9, init_loc=(1,1), goal_locs=[(9,9)], lava_locs=[()], gamma=0.99, slip_prob=0.00, name="four_room", is_goal_terminal=True, rand_init=False, lava_cost=0.01, step_cost=0.0):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
            lava_locs (list of tuples)
            gamma (float)
            slip_prob (float)
            name (str)
            is_goal_terminal (bool)
            rand_init (bool)
            lava_cost (float)
            step_cost (float)
        '''
        GridWorldMDP.__init__(self, width, height, init_loc, goal_locs=goal_locs, lava_locs=lava_locs, walls=self._compute_walls(width, height), gamma=gamma, slip_prob=slip_prob, name=name, is_goal_terminal=is_goal_terminal, rand_init=rand_init, lava_cost=lava_cost, step_cost=step_cost)

    def _compute_walls(self, width, height):
        '''
        Args:
            width (int)
            height (int)

        Returns:
            (list): Contains (x,y) pairs that define wall locations.
        '''
        walls = []

        half_width = math.ceil(width / 2.0)
        half_height = math.ceil(height / 2.0)

        # Wall from left to middle.
        for i in range(1, width + 1):
            if i == half_width:
                half_height -= 1
            if i + 1 == math.ceil(width / 3.0) or i == math.ceil(2 * (width + 2) / 3.0):
                continue

            walls.append((i, half_height))

        # Wall from bottom to top.
        for j in range(1, height + 1):
            if j + 1 == math.ceil(height / 3.0) or j == math.ceil(2 * (height + 2) / 3.0):
                continue
            walls.append((half_width, j))

        return walls
