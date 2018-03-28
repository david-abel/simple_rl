# Python imports
import numpy as np

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, data=[], is_terminal=False):
        self.data = data
        State.__init__(self, data=data, is_terminal=is_terminal)

    def to_rgb(self, x_dim, y_dim):
        # 3 by x_length by y_length array with values 0 (0) --> 1 (255)
        board = np.zeros(shape=[3, x_dim, y_dim])
        # print self.data, self.data.shape, x_dim, y_dim
        return self.data
