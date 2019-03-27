'''
PuddleMDPClass.py: Contains the Puddle class from:

    Boyan, Justin A., and Andrew W. Moore. "Generalization in reinforcement learning:
    Safely approximating the value function." NIPS 1995.
'''

# Python imports.
import math
import numpy as np
from collections import defaultdict

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class PuddleMDP(GridWorldMDP):
    ''' Class for a Puddle MDP '''

    def __init__(self, gamma=0.99, slip_prob=0.00, name="puddle", puddle_rects=[(0.1, 0.8, 0.5, 0.7), (0.4, 0.7, 0.5, 0.4)], goal_locs=[[1.0, 1.0]], is_goal_terminal=True, rand_init=False, step_cost=0.0):
        '''
        Args:
            gamma (float)
            slip_prob (float)
            name (str)
            puddle_rects (list): [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]
            is_goal_terminal (bool)
            rand_init (bool)
            step_cost (float)
        '''
        self.delta = 0.05
        self.puddle_rects = puddle_rects
        GridWorldMDP.__init__(self, width=1.0, height=1.0, init_loc=[0.25, 0.6], goal_locs=goal_locs, gamma=gamma, name=name, is_goal_terminal=is_goal_terminal, rand_init=rand_init, step_cost=step_cost)

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["slip_prob"] = self.slip_prob
        param_dict["is_goal_terminal"] = self.is_goal_terminal
        param_dict["rand_init"] = self.rand_init
        param_dict["puddle_rects"] = self.puddle_rects
   
        return param_dict

    def _reward_func(self, state, action, next_state):
        if state.is_terminal():
            return 0
        if self._is_goal_state_action(state, action):
            return 1.0 - self.step_cost
        elif self._is_puddle_state_action(state, action):
            return -1.0
        else:
            return 0 - self.step_cost

    def _is_puddle_state_action(self, state, action):
        '''
        Args:
            state (simple_rl.State)
            action (str)

        Returns:
            (bool)
        '''
        for puddle_rect in self.puddle_rects:
            x_1, y_1, x_2, y_2 = puddle_rect
            if state.x >= x_1 and state.x <= x_2 and \
                state.y <= y_1 and state.y >= y_2:
                return True

        return False

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        for g in self.goal_locs:
            if _euclidean_distance(state.x, state.y, g[0], g[1]) <= self.delta * 2 and self.is_goal_terminal:
                # Already at terminal.
                return False

        if action == "left" and self.is_loc_within_radius_to_goal(state.x - self.delta, state.y):
            return True
        elif action == "right" and self.is_loc_within_radius_to_goal(state.x + self.delta, state.y):
            return True
        elif action == "down" and self.is_loc_within_radius_to_goal(state.x, state.y - self.delta):
            return True
        elif action == "up" and self.is_loc_within_radius_to_goal(state.x, state.y + self.delta):
            return True
        else:
            return False

    def is_loc_within_radius_to_goal(self, state_x, state_y):
        '''
        Args:
            state_x (float)
            state_y (float)

        Returns:
            (bool)
        '''
        for g in self.goal_locs:
            if _euclidean_distance(state_x, state_y, g[0], g[1]) <= self.delta * 2:
                return True
        return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (simple_rl.State)
            action (str)

        Returns:
            state (simple_rl.State)
        '''
        if state.is_terminal():
            return state

        noise = np.random.randn(1)[0] / 100.0
        to_move = self.delta + noise

        if action == "up":
            next_state = GridWorldState(state.x, min(state.y + to_move, 1))
        elif action == "down":
            next_state = GridWorldState(state.x, max(state.y - to_move, 0))
        elif action == "right":
            next_state = GridWorldState(min(state.x + to_move, 1), state.y)
        elif action == "left":
            next_state = GridWorldState(max(state.x - to_move, 0), state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if self._is_goal_state_action(state, action) and self.is_goal_terminal:
            next_state.set_terminal(True)

        return next_state


def _euclidean_distance(ax, ay, bx, by):
    '''
    Args:
        ax (float)
        ay (float)
        bx (float)
        by (float)

    Returns:
        (float)
    '''
    return np.linalg.norm(np.array([ax, ay]) - np.array([bx, by]))
