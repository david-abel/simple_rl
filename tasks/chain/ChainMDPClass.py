# Python libs.
import random

from ChainStateClass import ChainState

class Chain(object):
	'''
	Imeplementation for an extension of NChain that also has a really bad (RMin) action in each state.
	'''
	def __init__(self, numStates=5):
		'''
		Args:
			numStates (int) [optional]: Number of states in the chain.
		'''
		self.numStates = numStates
		self.curState = DeadlyChainState(1)

	def getState(self):
		return self.curState

	def execPlayerAction(self, action):
		if action == "forward":
			if self.curState < self.numStates:
				# Move forward.
				self.curState += 1
				return 0
			elif self.curState.num == self.numStates:
				# Get reward.
				return 1
			else:
				print "Error: Unrecognized state! (forward action)", self.curState
				quit()
		elif action == "reset":
			# Go back to beginning.
			self.curState = DeadlyChainState(1)
			return 0.01
		elif action == "burn":
			# Burn.
			return -2
		else:
			print "Error: Unrecognized action!", action
			quit()

