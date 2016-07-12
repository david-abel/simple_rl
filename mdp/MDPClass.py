class MDP(object):
	''' Abstract class for a Markov Decision Process. '''
	def __init__(self, actions, transitionFunc, rewardFunc, initState, gamma=0.95):
		self.actions = actions
		self.transitionFunc = transitionFunc
		self.rewardFunc = rewardFunc
		self.gamma = gamma
		self.initState = initState
		self.curState = initState

	def getInitState(self):
		return self.initState

	def getCurrState(self):
		return self.curState

	def getActions(self):
		return self.actions

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