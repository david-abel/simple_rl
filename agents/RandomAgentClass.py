# Python libs.
import random

# Local libs.
from AgentClass import Agent

class RandomAgent(Agent):
	''' Class for a random decision maker. '''

	def __init__(self, actions):
		Agent.__init__(self, name="random",actions=actions)

	def act(self, state, reward):
		return random.choice(self.actions)