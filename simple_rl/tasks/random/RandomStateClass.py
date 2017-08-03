''' RandomStateClass.py: Contains the RandomStateClass class. '''

# Other imports
from simple_rl.mdp.StateClass import State

class RandomState(State):
    ''' Class for Random MDP States '''

    def __init__(self, num):
        State.__init__(self, data=num)
        self.num = num

    def __hash__(self):
        return self.num

    def __add__(self, val):
        return RandomState(self.num + val)

    def __lt__(self, val):
        return self.num < val

    def __str__(self):
        return "s." + str(self.num)

    def __eq__(self, other):
        '''
        Summary:
            Random states are equal when their num is the same
        '''
        return isinstance(other, RandomState) and self.num == other.num
