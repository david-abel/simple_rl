''' ChainStateClass.py: Contains the ChainStateClass class. '''

# Local imports.
from ...mdp.StateClass import State

class ChainState(State):
    ''' Class for Chain MDP States '''

    def __init__(self, num):
        State.__init__(self, data=num)
        self.num = num

    def __hash__(self):
        return self.num

    def __add__(self, val):
        return ChainState(self.num + val)

    def __lt__(self, val):
        return self.num < val

    def __str__(self):
        return "s." + str(self.num)

    def __eq__(self, other):
        '''
        Summary:
            Chain states are equal when their num is the same
        '''
        return isinstance(other, ChainState) and self.num == other.num
