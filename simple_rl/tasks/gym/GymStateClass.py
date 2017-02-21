# Python imports
import numpy

# Local imports
from ...mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, data=[], is_terminal=False):
        self.data = data
        State.__init__(self, data=data, is_terminal=is_terminal)
