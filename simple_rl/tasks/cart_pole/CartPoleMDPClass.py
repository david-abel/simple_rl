''' CartPoleMDPClass.py: Contains the CartPoleMDP class. '''

# Python imports.
from __future__ import print_function
import random, math
import sys, os, copy
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.cart_pole.CartPoleStateClass import CartPoleState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class CartPoleMDP(MDP):

    # forces (N) to exert on cart-pole system
    ACTIONS = [-10.0, 10.0] # negative for left, positive for right

    def __init__(self,
                gravity=9.8,
                masscart=1.0,
                masspole=0.1,
                length=.5,
                gamma=0.99,
                tau=.02,
                init_state_params=None,
                name="Cart-Pendulum"):

        if init_state_params is None:
            init_state = CartPoleState(x=0, x_dot=0, theta=0, theta_dot=0)
        else:
            init_state = CartPoleState(x=init_state_params["x"], x_dot=init_state_params["x_dot"],\
                                        theta=init_state_params["theta"], theta_dot=init_state_params["theta_dot"])

        MDP.__init__(self, CartPoleMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

        #from parameters
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.length = length
        self.gamma = gamma
        self.tau = tau
        self.name = name

        #thresholds
        self.x_threshold = 2.4 #abs val of limit of x position of cart
        self.theta_threshold = self._degrees_to_radians(20) #angle away from vertical before being considered terminal

        #computed
        self.total_mass = (self.masscart + self.masspole)
        self.polemass_length = (self.masspole * self.length)

    def _degrees_to_radians(self, degrees):
        return degrees * math.pi / 180

    #helper func
    def _is_within_threshold(self, theta, x):
        theta = theta % (2 * math.pi) #get theta value in (0,2pi) range
        val = theta if theta <= math.pi else (2 * math.pi - theta) #get angle away from vertical

        return True if (val < self.theta_threshold and abs(x) < self.x_threshold) else False

    #helper func 
    def _transition_helper(self, state, action):
        x, x_dot, theta, theta_dot = (state.x, state.x_dot, state.theta, state.theta_dot)
            
        force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - (self.masspole * costheta * costheta / self.total_mass)))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        #calculate the horizontal and angular velocities
        x_dot = x_dot + self.tau * xacc
        theta_dot = theta_dot + self.tau * thetaacc
        
        x = x + self.tau * x_dot
        theta = theta + self.tau * theta_dot
        
        return (x, x_dot, theta, theta_dot)

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)
        Returns
            (float)
        '''
        x, _, theta, _ = self._transition_helper(state, action)

        return 1.0 if self._is_within_threshold(theta, x) else -10.0


    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)
        Returns
            (State)
        '''
        x, x_dot, theta, theta_dot = self._transition_helper(state, action)
        next_state = CartPoleState(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

        #check if less than threshold values and if ternminal
        if not self._is_within_threshold(theta=next_state.theta, x=next_state.x):
            next_state.set_terminal(True)

        return next_state

    def __str__(self):
        return self.name #+ str(self.init_state) 

    def __repr__(self):
        return self.__str__()

    def reset(self, init_state_params=None):
        '''
        Args:
            init_state_params (dict)
        '''
        if init_state_params is None:
            self.init_state = copy.deepcopy(self.init_state)
        else:
            self.init_state = CartPoleState(x=init_state_params["x"], x_dot=init_state_params["x_dot"],\
                                        theta=init_state_params["theta"], theta_dot=init_state_params["theta_dot"])

        self.cur_state = self.init_state

def main():
    x = CartPoleMDP()

if __name__ == "__main__":
    main()