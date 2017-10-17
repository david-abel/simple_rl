''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class ColorState(GridWorldState):
    ''' Class for Color Grid World States '''

    def __init__(self, x, y, color):
        '''
        Args:
            x (int)
            y (int)
            color (int)
        '''
        State.__init__(self, data=[x, y, color])
        self.x = round(x, 3)
        self.y = round(y, 3)
        self.color = color

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ":" + str(self.color) + ")"

    def __eq__(self, other):
        return isinstance(other, ColorState) and self.x == other.x and self.y == other.y and self.color == other.color
