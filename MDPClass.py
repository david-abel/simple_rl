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
			(tuple: <float,State>): reward, State
		'''
		nextState = self.transitionFunc(self.curState, action)

		self.curState = nextState

		reward = self.rewardFunc(self.curState, action, nextState)

		return reward, nextState

	def reset(self):
		self.__init__()