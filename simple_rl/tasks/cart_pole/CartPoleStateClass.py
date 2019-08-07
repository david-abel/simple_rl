''' CartPoleStateClass.py: Contains the CartPoleState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State
import math

class CartPoleState(State):
    ''' Class for Cart Pole States '''
    def __init__(self, x, x_dot, theta, theta_dot):
        #using round to discretize each component of the state
        self.x = round(x, 1)
        self.x_dot = round(x_dot, 1)
        self.theta = round(theta, 3)
        self.theta_dot = round(theta_dot, 1)
        State.__init__(self, data=[self.x, self.x_dot, self.theta, self.theta_dot])
    
    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return " state: " + str(self.data)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, CartPoleState) and self.x == other.x and self.x_dot == other.x_dot and self.theta == other.theta and self.theta_dot == other.theta_dot