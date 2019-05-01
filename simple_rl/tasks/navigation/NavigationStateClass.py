# Other imports.
from simple_rl.mdp.StateClass import State


class NavigationWorldState(State):
    ''' Class for Navigation World States '''

    def __init__(self, x, y, phi=lambda state: [state.x, state.y]):
        State.__init__(self, data=[x, y])
        self.x = x
        self.y = y
        self.phi = phi

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (x = " + str(self.x) + ", y = " + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return isinstance(other, NavigationWorldState) and \
            self.x < other.x and self.y < other.y

    def __eq__(self, other):
        return isinstance(other, NavigationWorldState) and \
            self.x == other.x and self.y == other.y

    def __getitem__(self, index):
        return self.data[index]

    def features(self):
        return self.phi(self)
