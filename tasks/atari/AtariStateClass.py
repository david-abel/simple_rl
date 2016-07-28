''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Local libs.
from simple_rl.mdp.ImageStateClass import ImageState

class AtariState(ImageState):
    ''' Class for Atari States '''

    def __init__(self, image, ram, terminal=False):
        ImageState.__init__(self, image=image, state_id=ram)
        self._is_terminal = terminal

    def is_terminal(self):
        return self._is_terminal