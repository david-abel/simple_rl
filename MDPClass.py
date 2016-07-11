class MDP(object):
	''' Abstract class for a Markov Decision Process. '''
	def __init__(self, actions, transitionFunc, rewardFunc, gamma=0.95, initState=None):
		self.actions = actions
		self.transitionFunc = transitionFunc
		self.rewardFunc = rewardFunc
		self.gamma = gamma
		self.curState = initState

	def getCurState(self):
		return self.curState

	def executeAgentAction(self, action):
		'''
		Args:
			action (str)

		Returns:
			(tuple: <State,float>)
		'''
		nextState = self.transitionFunc(self.curState, action)

		reward = self.rewardFunc(self.curState, action, nextState)

		return nextState, reward

