class GraphMDP(MDP):

	def __init__(self, states, actions, transition_matrix, reward_matrix):
		'''
		Args:
			states
			actions
			transition_matrix_a (dict: key=a, val="matrix")
			reward_matrix_a (dict: key=a, val="matrix")
		'''
		self.states = states
		self.actions = actions
		self.transition_matrix_a = transition_matrix_a
		self.reward_matrix_a = self.reward_matrix_a

	def _transition_func(self, state, action):
		self.transition_matrix_a[a][]