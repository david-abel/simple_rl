# Python imports.
import sys
import os
sys.path.append(os.getcwd() + "/..")

# Local imports.
from MDPClass import MDP
from GridWorldStateClass import GridWorldState

class GridWorld(MDP):
	''' Class for a Grid World MDP '''
	
	# Static constants.
	actions = ["up", "down", "left", "right"]

	def __init__(self, height, width, initLoc, goalLoc):
		MDP.__init__(self, GridWorld.actions, self._transitionFunction, self._rewardFunction)		
		self.height = height
		self.width = width
		self.curState = GridWorldState(initLoc[0], initLoc[1])
		self.goalState = GridWorldState(goalLoc[0], goalLoc[1])

	def _rewardFunction(self, state, action, statePrime):
		'''
		Args:
			state (State)
			action (str)
			statePrime

		Returns
			float
		'''

		if statePrime == self.goalState:
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
		if action not in GridWorld.actions:
			print "Error: the action provided (" + str(action) + ") was invalid."
			quit()

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

	def __str__(self):
		return "gridworld(h:" + str(self.height) + ",w:" + str(self.width) + ")"



def main():
	gw = GridWorld(5,10, (1,1), (6,7))

if __name__ == "__main__":
	main()