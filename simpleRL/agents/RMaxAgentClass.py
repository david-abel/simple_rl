# Local classes.
from AgentClass import Agent

# Python libs.
import random
import numpy
from collections import defaultdict

class RMaxAgent(Agent):
    ''' Implementation for an R-Max Agent [Brafman and Tennenholtz 2003] '''

    ''' NOTE: Assumes WLOG R \in [0,1] (so RMAX is 1.0) '''

    def __init__(self, actions, gamma=0.95):
        Agent.__init__(self, name="rmax", actions=actions, gamma=gamma)
        self.rMax = 1.0
        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.R = defaultdict(lambda : self.rMax) # keys are (s,a) pairs, value is int.
        self.T = defaultdict(lambda : None) # key is (s,a) pair, val is a state (DETERMINISTIC for now)
        self.stateCounts = defaultdict(int) # key is (s,a) pair, value is int (default 0).
        self.prevState = None
        self.prevAction = None

    def act(self, state, reward):
        # s,a,r,s' : self.prevState, self.prevAction, reward, state
        self.stateCounts[(self.prevState, self.prevAction)] += 1
        self.R[(self.prevState, self.prevAction)] = reward
        self.T[(self.prevState, self.prevAction)] = state

        # Compute best action.
        action = self.getMaxQAction(state)
        self.prevAction = action
        self.prevState = state

        return action

    def getMaxQAction(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): the string associated with the action with highest Q value.
        '''
        maxQ = float("-inf")
        bestAction = None

        # Find Max Q Action
        for action in self.actions:
            Q_s_a = self.getQValue(state, action)

            if Q_s_a > maxQ:
                maxQ = Q_s_a
                bestAction = action

        return bestAction

    def computeQValueOfState(self, state, horizon=5):
        '''
        Args:
            state (State)

        Returns:
            (float): max Q value for this state
        '''

        if state == None:
            return self.rMax * horizon

        maxQ = float("-inf")
        for action in self.actions:
            Q_s_a = self.getQValue(state, action, horizon)
            if Q_s_a > maxQ:
                maxQ = Q_s_a
        return maxQ

    def getQValue(self, state, action, horizon=5):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): indicates the level of recursion depth for computing Q.

        Returns:
            (float)
        '''

        # If we've hashed a Q value for this already.
        if (state,action) in self.Q:
            return self.Q[(state,action)]

        if horizon == 0:
            return self.R[(state,action)]

        nextState = self.T[(state, action)]

        q = self.R[(state, action)] + self.computeQValueOfState(nextState, horizon = horizon-1)

        return q
