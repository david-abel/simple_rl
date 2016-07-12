# Python imports.
import sys
import os
import random

# Local imports.
from mdp.MDPClass import MDP
from mdp.StateClass import State
from GridWorldStateClass import GridWorldState

class GridWorldMDP(MDP):
	''' Class for a Grid World MDP '''
	
	# Static constants.
	actions = ["up", "down", "left", "right"]

	def __init__(self, height, width, initLoc, goalLoc):
		MDP.__init__(self, GridWorldMDP.actions, self._transitionFunction, self._rewardFunction, initState = GridWorldState(initLoc[0], initLoc[1]))		
		self.height = height
		self.width = width
		self.initLoc = initLoc
		self.goalLoc = goalLoc
		self.curState = GridWorldState(initLoc[0], initLoc[1])
		self.goalState = GridWorldState(goalLoc[0], goalLoc[1])

	def _rewardFunction(self, state, action, statePrime):
		'''
		Args:
			state (State)
			action (str)
			statePrime (State)

		Returns
			(float)
		'''
		self._errorCheck(state, action, statePrime)

		if statePrime.x == self.goalLoc[0] and statePrime.y == self.goalLoc[1]:
			print "Goal"
			return 1
		else:
			return -0.5

	def _transitionFunction(self, state, action):
		'''
		Args:
			state (State)
			action (str)

		Returns
			(State)
		'''
		self._errorCheck(state, action)

		if action == "up" and state.y < self.height:
			return GridWorldState(state.x, state.y + 1)
		elif action == "down" and 1 < state.y:
			return GridWorldState(state.x, state.y - 1)
		elif action == "right" and state.x < self.width:
			return GridWorldState(state.x + 1, state.y)
		elif action == "left" and 1 < state.x:
			return GridWorldState(state.x - 1, state.y)
		else:
			return GridWorldState(state.x, state.y)

	def _errorCheck(self, state, action, sPrime = -1):
		'''
		Args:
			state (State)
			action (str)
			sPrime (State): -1 if left blank (and thus type is int).

		Summary:
			Checks to make sure we've received states/actions of the right type.
		'''
		
		if action not in GridWorldMDP.actions:
			print "Error: the action provided (" + str(action) + ") was invalid."
			quit()

		if not(isinstance(state, GridWorldState)):
			print "Error: the given state (" + str(state) + ") was not of the correct class."
			quit()

		if not(sPrime == -1 or isinstance(state, GridWorldState)):
			print "Error: the given state (" + str(sPrime) + ") was not of the correct class."
			quit()


	def __str__(self):
		return "gridworld_h-" + str(self.height) + "_w-" + str(self.width)


def main():
	gw = GridWorld(5,10, (1,1), (6,7))

if __name__ == "__main__":
	main()