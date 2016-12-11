''' AgentClass.py: Class for a basic RL Agent '''

# Python imports.
from collections import defaultdict

class Agent(object):
    ''' Abstract Agent class. '''

    def __init__(self, name, actions, gamma=0.95):
        self.name = name
        self.actions = list(actions) # Just in case we're given a numpy array (like from Atari).
        self.gamma = gamma

        self.prev_state = None
        self.prev_action = None
        self.default_q = 0.0
        self.q_func = defaultdict(lambda: self.default_q)

    def act(self, state, reward):
        '''
        Args:
            state (State): see StateClass.py
            reward (float): the reward associated with arriving in state @state.

        Returns:
            (str): action.
        '''
        pass

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.prev_state = None
        self.prev_action = None
        self.q_func = defaultdict(lambda: self.default_q)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        self.prev_state = None
        self.prev_action = None

    def __str__(self):
        return str(self.name)
