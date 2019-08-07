# Python imports.
from collections import defaultdict
import random

# Other imports.
from simple_rl.pomdp.POMDPClass import POMDP
from simple_rl.tasks.maze_1d.Maze1DStateClass import Maze1DState

class Maze1DPOMDP(POMDP):
    ''' Class for a 1D Maze POMDP '''

    ACTIONS = ['west', 'east']
    OBSERVATIONS = ['nothing', 'goal']

    def __init__(self):
        self._states = [Maze1DState('left'), Maze1DState('middle'), Maze1DState('right'), Maze1DState('goal')]

        # Initial belief is a uniform distribution over states
        b0 = defaultdict()
        for state in self._states: b0[state] = 0.25

        POMDP.__init__(self, Maze1DPOMDP.ACTIONS, Maze1DPOMDP.OBSERVATIONS, self._transition_func, self._reward_func, self._observation_func, b0)

    def _transition_func(self, state, action):
        '''
        Args:
            state (Maze1DState)
            action (str)

        Returns:
            next_state (Maze1DState)
        '''
        if action == 'west':
            if state.name == 'left':
                return Maze1DState('left')
            if state.name == 'middle':
                return Maze1DState('left')
            if state.name == 'right':
                return Maze1DState('goal')
            if state.name == 'goal':
                return Maze1DState(random.choice(['left', 'middle', 'right']))
        if action == 'east':
            if state.name == 'left':
                return Maze1DState('middle')
            if state.name == 'middle':
                return Maze1DState('goal')
            if state.name == 'right':
                return Maze1DState('right')
            if state.name == 'goal':
                return Maze1DState(random.choice(['left', 'middle', 'right']))
        raise ValueError('Invalid state: {} action: {} in 1DMaze'.format(state, action))

    def _observation_func(self, state, action):
        next_state = self._transition_func(state, action)
        return 'goal' if next_state.name == 'goal' else 'nothing'

    def _reward_func(self, state, action, next_state):
        # next_state = self._transition_func(state, action)
        observation = self._observation_func(state, action)
        return (1. - self.step_cost) if (next_state.name == observation == 'goal') else (0. - self.step_cost)

    def is_in_goal_state(self):
        return self.cur_state.name == 'goal'

if __name__ == '__main__':
    maze_pomdp = Maze1DPOMDP()
