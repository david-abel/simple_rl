
class Agent(object):
	''' Abstract Agent class. '''

	def __init__(self, name, actions, gamma=0.95):
		self.name = name
		self.actions = actions
		self.gamma = gamma

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
		pass

	def __str__(self):
		return str(self.name)