''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Local libs.
from simple_rl.mdp.StateClass import State

class GridWorldState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y):
        State.__init__(self, data=[x, y])
        self.x = x
        self.y = y

    def __hash__(self):
        return int(str(self.x) + "0990" + str(self.y))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ")"

    def __eq__(self, other):
        return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y
