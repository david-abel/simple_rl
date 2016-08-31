''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Local libs.
from simple_rl.mdp.ImageStateClass import ImageState
from simple_rl.mdp.oomdp.OOMDPImageStateClass import OOMDPImageState

class AtariState(OOMDPImageState, ImageState):
    ''' Class for Atari States '''

    def __init__(self, image, ram_data, terminal=False, objects_from_image=False):
        if objects_from_image:
            # Create the state as an OOMDP State.
            # --> Extracts objects using color segmentation on the image.
            OOMDPImageState.__init__(self, image=image)
        else:
            # Create the state based on the raw image/ram.
            self.state_id = _compute_state_id(ram_data)
            ImageState.__init__(self, image=image, features=ram_data)

        self._is_terminal = terminal

    def is_terminal(self):
        return self._is_terminal

def _compute_state_id(ram_data):
    '''
    Args:
        ram_data (1d numpy matrix)

    Returns:
        (int): turns the matrix into an int.
    '''
    result = ""
    # Convert every value to a 3 digit string.
    for ram_val in ram_data:
        ram_str = ""
        i = 0
        while len(ram_str) < 3:
            if i < len(str(ram_val)):
                ram_str += str(ram_val)[i]
            else:
                ram_str = "0" + ram_str
            i += 1
        result += ram_str
    return int(result)