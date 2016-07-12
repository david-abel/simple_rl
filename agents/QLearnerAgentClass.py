# Misc. python libs
import numpy as np
import random
from collections import defaultdict

# Local classes
from AgentClass import Agent

class QLearnerAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, actions, alpha=0.05, gamma=0.95, epsilon=0.01):
        '''
        Args:
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
        '''
        Agent.__init__(self, name="qlearner", actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha = alpha
        self.epsilon = epsilon

        self.prevState = None
        self.prevAction = None
        self.defaultQ = 0.0
        self.Q = defaultdict(lambda : self.defaultQ)

        # Choose explore type. Can also be "uniform" for \epsilon-greedy.
        self.explore = "softmax"

    def reset(self):
        self.prevState = None
        self.prevAction = None
        self.Q = defaultdict(lambda : self.defaultQ)

    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward):
        '''
        Args:
            state (State)
            reward (float)

        Summary:
            The central method called during each time step. Retrieves the action according to the current policy and performs updates given (s=self.prevState,a=self.prevAction,r=reward,s'=state)
        '''
        self.updateQ(state, reward)

        if self.explore == "softmax":
            # Softmax exploration
            action = self.softMaxPolicy(state)
        else:
            # Uniform exploration
            action = self.epsilonGreedyQPolicy(state)
            
        self.prevState = state
        self.prevAction = action
        return action

    def epsilonGreedyQPolicy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): action.
        '''
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if np.random.random() > self.epsilon:
            # Exploit.
            action = self.getMaxQAction(state)
        else:
            # Explore
            action = np.random.choice(self.actions)

        return action

    def softMaxPolicy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.

        Returns:
            (str): action.
        '''
        return np.random.choice(self.actions, 1, p=self.getActionDistr(state))[0]

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

    def updateQ(self, currState, reward):
        '''
        Args:
            currState (State): A State object containing the abstracted state representation
            reward (float): The real valued reward of the associated state

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        # If this is the first state, just return.
        if self.prevState == None:
            self.prevState = currState
            return

        # Update the Q Function.
        maxQCurrState = self.getMaxQValue(currState)
        prevQVal = self.getQValue(self.prevState , self.prevAction)
        self.Q[(self.prevState, self.prevAction)] = (1 - self.alpha) * prevQVal + self.alpha * (reward + self.gamma*maxQCurrState)

    def _computeMaxQValActionPair(self, state):
        ''' 
        Args:
            state (State)

        Returns:
            (tuple) --> (float,str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        bestAction = None
        maxQVal = float("-inf")
        shuffledActionList = self.actions[:]
        random.shuffle(shuffledActionList)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffledActionList:
            Q_s_a = self.getQValue(state, action)
            if Q_s_a > maxQVal:
                maxQVal = Q_s_a
                bestAction = action

        return maxQVal, bestAction

    def getMaxQAction(self, state):
        ''' 
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._computeMaxQValActionPair(state)[1]

    def getMaxQValue(self, state):
        ''' 
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._computeMaxQValActionPair(state)[0]



    def getQValue(self , state , action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state,@action) pair.
        '''
        return self.Q[(state, action)]

    def getActionDistr(self, state):
        '''
        Args:
            state (State)

        Returns:
            (list of floats): The i-th float corresponds to the probability mass associated with the i-th action (indexing into self.actions)
        '''
        all_q_vals = []
        for i in xrange(len(self.actions)):
            action = self.actions[i]
            all_q_vals.append(self.getQValue(state, action))

        # Softmax distribution.
        total = sum([np.exp(qv) for qv in all_q_vals])
        softmax = [np.exp(qv) / total for qv in all_q_vals]

        return softmax
