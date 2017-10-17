''' RockSampleMDPClass.py: Contains the RockSample class. '''

# Python imports.
import random
import math

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.mdp.StateClass import State

class RockSampleMDP(GridWorldMDP):
    '''
        Class an MDP adaption of the RockSample POMDP from:

            Trey Smith and Reid Simmons: "Heuristic Search Value Iteration for POMDPs" UAI 2004.
    '''

    ACTIONS = ["up", "down", "left", "right", "sample"]

    def __init__(self, width=8, height=7, init_loc=(1,1), rocks=[[1,2,True], [3,1,True], [4,2,True], [3,5,False], [4,5,True], [2,7,True], [6,6,True], [7,4,False]], gamma=0.99, slip_prob=0.00, rock_rewards=[0.1, 1, 20], name="rocksample"):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        self.init_loc = init_loc
        self.init_rocks = rocks[:]
        self.rocks = rocks
        self.rock_rewards = rock_rewards
        self.name = name + "-" + str(len(rocks))
        self.width = width
        self.height = height
        MDP.__init__(self, RockSampleMDP.ACTIONS, self._transition_func, self._reward_func, init_state=self.get_init_state(), gamma=gamma)

    def make_state(self, x, y):
        features = [x, y]
        for rock in self.rocks:
            int_rock = [int(f) for f in rock]
            features += list(int_rock)

        return State(data=features)

    def get_init_state(self):
        return self.make_state(self.init_loc[0], self.init_loc[1])

    def _reward_func(self, state, action):
        if state[0] == 7 and action == "right":
            # Moved into exit area, receive 10 reward.
            return 10.0
        elif action == "sample":
            rock_index = self._get_rock_at_agent_loc(state)
            if rock_index != None:
                if self.rocks[rock_index][2]:
                    # Sampled good rock.
                    return self.rock_rewards[rock_index % 3]
                else:
                    # Sampled bad rock.
                    return -self.rock_rewards[rock_index % 3]

        return 0

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state

        if action == "sample":
            # Sample action.
            rock_index = self._get_rock_at_agent_loc(state)
            if rock_index != None:
                # Set to false.
                self.rocks[rock_index][2] = False
                next_state = self.make_state(state[0], state[1])
            else:
                next_state = self.make_state(state[0], state[1])
        elif action == "up" and state[1] < self.height:
            next_state = self.make_state(state[0], state[1] + 1)
        elif action == "down" and state[1] > 1:
            next_state = self.make_state(state[0], state[1] - 1)
        elif action == "right" and state[0] < self.width:
            next_state = self.make_state(state[0] + 1, state[1])
        elif action == "left" and state[0] > 1:
            next_state = self.make_state(state[0] - 1, state[1])
        else:
            next_state = self.make_state(state[0], state[1])

        if next_state[0] > 7:
            next_state.set_terminal(True)

        return next_state

    def _get_rock_at_agent_loc(self, state):
        result = None

        for i, r in enumerate(self.rocks):
            if r[0] == state[0] and r[1] == state[1]:
                return i

        # No rock found.
        return None

    def __str__(self):
        return self.name

    def _reset(self):
        self.rocks = init_rocks[:]
        GridWorldMDP._reset(self)