''' RandomAgentClass.py: Class for a randomly acting RL Agent '''

# Python libs.
import random

# Local libs.
from simple_rl.agents.AgentClass import Agent

class RandomAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, actions):
        Agent.__init__(self, name="random", actions=actions)

    def act(self, state, reward):
        return random.choice(self.actions)
