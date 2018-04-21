class Policy(object):

	def __init__(self, policy_lambda):
		self.policy_lambda = policy_lambda

	def get_action(self, state):
		return self.policy_lambda(state)
