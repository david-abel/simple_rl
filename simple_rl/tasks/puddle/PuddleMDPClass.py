'''
PuddleMDPClass.py: Contains the Puddle class from:

    Boyan, Justin A., and Andrew W. Moore. "Generalization in reinforcement learning:
    Safely approximating the value function." NIPS 1995.
'''

# Python imports.
import math

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class PuddleMDP(GridWorldMDP):
    ''' Class for a Puddle MDP '''

    def __init__(self, gamma=0.99, slip_prob=0.00, name="puddle", is_goal_terminal=True, rand_init=False):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        GridWorldMDP.__init__(self, width=1.0, height=1.0, init_loc=[0.0,0.0], goal_locs=[1.0,1.0], gamma=gamma, name=name, is_goal_terminal=is_goal_terminal, rand_init=rand_init)

    def _reward_func(self, state, action):


    def _transition_func(self, state, action):

        if action == "up":
            next_state = GridWorldState(state.x, state.y + .01)
        elif action == "down":
            next_state = GridWorldState(state.x, state.y - .01)
        elif action == "right":
            next_state = GridWorldState(state.x + .01, state.y)
        elif action == "left":
            next_state = GridWorldState(state.x - .01, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) in self.goal_locs and self.is_goal_terminal:
            next_state.set_terminal(True)


        return next_state



