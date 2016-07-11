# Misc. python libs
from collections import defaultdict
import numpy as np
import random

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
        Agent.__init__(self, name="qlearner", actions=actions)

        # Set/initialize parameters and other relevant classwide data
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.prevState = None
        self.prevAction = None
        self.Q = defaultdict(float)

        self.explore = "softmax"

    def reset(self):
        self.prevState = None
        self.prevAction = None
        self.Q = defaultdict(float)

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

        # Update Q Function according to the Bellman Equation.
        maxQCurrState = self.getMaxQValue(currState)
        prevQVal = self.getQValue(self.prevState , self.prevAction)
        self.Q[(self.prevState, self.prevAction)] = (1 - self.alpha) * prevQVal + self.alpha * (reward + self.gamma*maxQCurrState)

    def getMaxQAction(self, state):
        ''' 
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
    
        # Grab random initial action in case all equal
        best_action = None
        max_qval = float("-inf")
        shuffledActionList = self.actions[:]
        random.shuffle(shuffledActionList)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffledActionList:
            Q_s_a = self.getQValue(state, action)
            if Q_s_a > max_qval:
                max_qval = Q_s_a
                best_action = action

        return best_action

    def getMaxQValue(self, state):
        ''' 
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
    
        max_qval = float("-inf")

        # Find best action (action w/ current max predicted Q value).
        for action in self.actions:
            Q_s_a = self.getQValue(state, action)
            if Q_s_a > max_qval:
                max_qval = Q_s_a

        return max_qval


    def getQValue(self , state , action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state,@action) pair.
        '''
        if (state, action) not in self.Q:
            # If the Q value isn't in there yet, initialize it to the default before returning.
            self.Q[(state, action)] = 0
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
