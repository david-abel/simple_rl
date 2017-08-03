''' GridGameStateClass.py: Contains the GridGameState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class GridGameState(State):
    ''' Class for two player Grid Game States '''

    def __init__(self, a_x, a_y, b_x, b_y):
        State.__init__(self, data=[a_x, a_y, b_x, b_y])
        self.a_x = a_x
        self.a_y = a_y
        self.b_x = b_x
        self.b_y = b_y

    def __hash__(self):
        # The X coordinate takes the first three digits.
        if len(str(self.a_x)) < 3:
            a_x_str = str(self.a_x)
            while len(a_x_str) < 3:
                a_x_str = "0" + a_x_str

        # The Y coordinate takes the next three digits.
        if len(str(self.a_y)) < 3:
            a_y_str = str(self.a_y)
            while len(a_y_str) < 3:
                a_y_str = "0" + a_y_str

        # The X coordinate takes the first three digits.
        if len(str(self.b_x)) < 3:
            b_x_str = str(self.b_x)
            while len(b_x_str) < 3:
                b_x_str = "0" + b_x_str

        # The Y coordinate takes the next three digits.
        if len(str(self.b_y)) < 3:
            b_y_str = str(self.b_y)
            while len(b_y_str) < 3:
                b_y_str = "0" + b_y_str

        # Concatenate and return.
        return int(a_x_str + a_y_str + "0" + b_x_str + b_y_str)

    def __str__(self):
        return "s: (" + str(self.a_x) + "," + str(self.a_y) + ")_a (" + str(self.b_x) + "," + str(self.b_y) + ")_b"

    def __eq__(self, other):
        return isinstance(other, GridGameState) and self.a_x == other.a_x and self.a_y == other.a_y and \
            self.b_x == other.b_x and self.b_y == other.b_y
