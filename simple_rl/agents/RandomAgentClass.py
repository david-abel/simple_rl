''' RandomAgentClass.py: Class for a randomly acting RL Agent '''

# Python imports.
import random

# Local imports.
from AgentClass import Agent

class RandomAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, actions, name=""):
    	name = "random" if name is "" else name
        Agent.__init__(self, name=name, actions=actions)

    def act(self, state, reward):
        return random.choice(self.actions)
