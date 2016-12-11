''' AtariStateClass.py: Contains the AtariState class. '''

# Local imports.
from ...mdp.ImageStateClass import ImageState
from ...mdp.oomdp.OOMDPImageStateClass import OOMDPImageState

class AtariState(ImageState):
    ''' Class for Atari States: Creates the state based on the raw image/ram. '''

    def __init__(self, image, ram_data, terminal=False):
        ImageState.__init__(self, image=image, features=ram_data)
        self._is_terminal = terminal

    def is_terminal(self):
        return self._is_terminal