''' ColorMDPClass.py: Contains the Color class. '''

# Python imports.
import random
import math

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.tasks.color.ColorStateClass import ColorState

class ColorMDP(GridWorldMDP):
    ''' Class for a Color '''

    def __init__(self, width=9, height=9, rand_init=False, is_four_room=False, num_colors=5, init_loc=(1,1), goal_locs=[(9,9)], gamma=0.99, slip_prob=0.00, name="color"):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        self.num_colors = num_colors
        if is_four_room:
            walls = self._compute_walls(width, height)
        else:
            walls = []
        init_state = ColorState(init_loc[0], init_loc[1], color=random.randint(1, self.num_colors))
        GridWorldMDP.__init__(self, width, height, init_loc, rand_init=rand_init, init_state=init_state, goal_locs=goal_locs, walls=walls, gamma=gamma, slip_prob=slip_prob, name=str(self.num_colors) + name)

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

        for i in range(1, width + 1):
            if i == (width + 1) / 3 or i == math.ceil(2 * (width + 1) / 3.0):
                continue
            walls.append((i, half_height))

        for j in range(1, height + 1):
            if j == (height + 1) / 3 or j == math.ceil(2 * (height + 1) / 3.0):
                continue
            walls.append((half_width, j))

        return walls

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''

        gw_state = GridWorldState(state.x, state.y)

        next_gw_state = GridWorldMDP._transition_func(self, gw_state, action)

        # Add random color.
        rand_color = random.randint(1, self.num_colors)
        next_col_state = ColorState(next_gw_state.x, next_gw_state.y, rand_color)

        return next_col_state