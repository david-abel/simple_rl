''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Local imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

# Python imports.
import random

class GridWorldMDP(MDP):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, height, width, init_loc, goal_loc):
        MDP.__init__(self, GridWorldMDP.ACTIONS, self._transition_func, self._reward_func, init_state=GridWorldState(init_loc[0], init_loc[1]))
        self.height = height
        self.width = width
        self.init_loc = init_loc
        self.goal_loc = goal_loc
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        self.goal_state = GridWorldState(goal_loc[0], goal_loc[1])

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        _error_check(state, action)

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
        if action == "left" and (state.x == self.goal_state.x + 1) \
            or (state.x == 1 == self.goal_state.x):
            return True
        elif action == "right" and (state.x == self.goal_state.x + 1) \
            or (state.x == self.width == self.goal_state.x):
            return True
        elif action == "down" and (state.y == self.goal_state.y - 1) \
            or (state.y == 1 == self.goal_state.y):
            return True
        elif action == "up" and (state.y == self.goal_state.y + 1) \
            or (state.y == self.height == self.goal_state.y):
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
        _error_check(state, action)

        if action == "up" and state.y < self.height:
            return GridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1:
            return GridWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width:
            return GridWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1:
            return GridWorldState(state.x - 1, state.y)
        else:
            return GridWorldState(state.x, state.y)

    def __str__(self):
        return "gridworld_h-" + str(self.height) + "_w-" + str(self.width)


def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in GridWorldMDP.ACTIONS:
        print "Error: the action provided (" + str(action) + ") was invalid."
        quit()

    if not isinstance(state, GridWorldState):
        print "Error: the given state (" + str(state) + ") was not of the correct class."
        quit()



def main():
    grid_world = GridWorldMDP(5, 10, (1, 1), (6, 7))

if __name__ == "__main__":
    main()
