''' ChainStateClass.py: Contains the ChainStateClass class. '''

# Local imports.
from ...mdp.StateClass import State

class FruitbowlState(State):
    ''' Class for Fruitbowl MDP States '''

    def __init__(self, bowl_state="", index=0):
        State.__init__(self, data=[bowl_state, index])
        self.index = index
        self.bowl_state = bowl_state
