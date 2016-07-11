
class Agent(object):
	'''
	Abstract Agent class.
	'''

	def __init__(self, name, actions):
		self.name = name
		self.actions = actions

	def policy(self, state):
		'''
		Args:
			state (State): see StateClass.py

		Returns:
			(str): action.
		'''
		pass

	def __str__(self):
		return str(self.name)