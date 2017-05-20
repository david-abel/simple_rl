''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Local imports.
from ...mdp.StateClass import State

class GridWorldState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y):
        State.__init__(self, data=[x, y])
        self.x = x
        self.y = y

    def __hash__(self):
        # The X coordinate takes the first three digits.
        if len(str(self.x)) < 3:
            x_str = str(self.x)
            while len(x_str) < 3:
                x_str = "0" + x_str

        # The Y coordinate takes the next three digits.
        if len(str(self.y)) < 3:
            y_str = str(self.y)
            while len(y_str) < 3:
                y_str = "0" + y_str

        # Concatenate and return.
        return int(x_str + y_str)

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ")"

    def __eq__(self, other):
        return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y
