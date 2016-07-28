''' StateClass.py: Contains the State Class. '''

# Python imports.
import numpy

# Local libs.
from simple_rl.mdp.StateClass import State

class ImageState(State):
    ''' Abstract State class '''

    def __init__(self, image, state_id=0):
    	'''
    	Args:
    		image (np matrix)
    		state_id (int)
    	'''
        State.__init__(self, data=[state_id])
        self.image = image
        self.state_id = state_id

    def __eq__(self, other):
    	return self.state_id == other.state_id

    def __hash__(self):
    	return self.state_id