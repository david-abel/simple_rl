# Python libs. 
import sys
import os

# Local libs.
from simpleRL.mdp.StateClass import State

class GridWorldState(State):
	''' Class for Grid World States '''

	def __init__(self, x, y, hasAgent = False):
		self.x = x
		self.y = y

	def __hash__(self):
		return int(str(self.x) + "0990" + str(self.y))

	def __str__(self):
		return "s: (" + str(self.x) + "," + str(self.y) + ")"

	def __eq__(self, other):
		'''
		Summary:
			GridWorld states are equal when their x and y are the same.
		'''
		if isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y:
			return True
		else:
			return False