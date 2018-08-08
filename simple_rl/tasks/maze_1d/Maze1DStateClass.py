from simple_rl.mdp.StateClass import State

class Maze1DState(State):
    ''' Class for 1D Maze POMDP States '''

    def __init__(self, name):
        self.name = name
        is_terminal = name == 'goal'
        State.__init__(self, data=name, is_terminal=is_terminal)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return '1DMazeState::{}'.format(self.data)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Maze1DState) and self.data == other.data
