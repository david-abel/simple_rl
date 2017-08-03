''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class GridWorldState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y):
        State.__init__(self, data=[x, y])
        self.x = x
        self.y = y

    def __hash__(self):
        # The X coordinate takes the first chunk of digits.
        x_str = "0" + str(self.x)

        # The Y coordinate takes the next chunk of digits.
        y_str = "0" + str(self.x)

        # Concatenate and return.
        return int(x_str + y_str)

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ")"

    def __eq__(self, other):
        # print self.x, other.x, self.y, other.y, isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y
        return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y
