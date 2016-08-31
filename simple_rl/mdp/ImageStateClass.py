''' StateClass.py: Contains the State Class. '''

# Python imports.
import numpy

# Local libs.
from simple_rl.mdp.StateClass import State

class ImageState(State):
    ''' Abstract State class '''

    def __init__(self, image, features=None):
        '''
        Args:
            image (np matrix)
            features (1d numpy matrix)
        '''
        self.image = image
        State.__init__(self, data=features)

    def get_image(self):
        return self.image
