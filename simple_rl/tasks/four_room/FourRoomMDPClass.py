''' FourRoomMDPClass.py: Contains the FourRoom class. '''

# Local imports.
from ...mdp.MDPClass import MDP
from ..grid_world.GridWorldStateClass import GridWorldState

# Python imports.
import random
import math

class FourRoomMDP(MDP):
    ''' Class for a FourRoom '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, width=6, height=6, init_loc=(1,1), goal_loc=(6,6)):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        MDP.__init__(self, FourRoomMDP.ACTIONS, self._transition_func, self._reward_func, init_state=GridWorldState(init_loc[0], init_loc[1]))

        self.width = width
        self.height = height
        self.init_loc = init_loc
        self.goal_loc = goal_loc
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])

        self.walls = self._compute_walls()

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if self._is_goal_state_action(state, action):
            return 1
        else:
            return 0

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if action == "left" and (state.x - 1, state.y) == self.goal_loc:
            return True
        elif action == "right" and (state.x + 1, state.y) == self.goal_loc:
            return True
        elif action == "down" and (state.x, state.y - 1) == self.goal_loc:
            return True
        elif action == "up" and (state.x, state.y + 1) == self.goal_loc:
            return True
        else:
            return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if action == "up" and state.y < self.height and not self.is_wall(state.x, state.y + 1):
            next_state = GridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1 and not self.is_wall(state.x, state.y - 1):
            next_state = GridWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width and not self.is_wall(state.x + 1, state.y):
            next_state = GridWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1 and not self.is_wall(state.x - 1, state.y):
            next_state = GridWorldState(state.x - 1, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) == self.goal_loc:
            next_state.set_terminal(True)

        return next_state

    def __str__(self):
        return "four_rooms_h-" + str(self.height) + "_w-" + str(self.width)

    def is_wall(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''
        return (x, y) in self.walls

    def _compute_walls(self):
        '''
        Returns:
            (list): Contains (x,y) pairs that define wall locations.
        '''
        walls = []

        half_width = self.width / 2
        half_height = self.height / 2

        for i in range(1, self.width + 1):
            if i == self.width / 3 or i == math.ceil(2 * self.width / 3.0):
                continue
            walls.append((i, half_height))

        for j in range(1, self.height + 1):
            if j == self.height / 3 or j == math.ceil(2 * self.height / 3.0):
                continue
            walls.append((half_width, j))

        return walls
