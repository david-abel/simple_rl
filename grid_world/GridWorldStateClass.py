# Python libs. 
import sys
import os

# Setup path, import relevant libs.
os.chdir(os.path.dirname(__file__))
sys.path.append(os.getcwd() + "../")

# Local libs.
from StateClass import State

class GridWorldState(State):
	''' Class for Grid World States '''

	def __init__(self, x, y, hasAgent = False):
		self.x = x
		self.y = y

	def __str__(self):
		return "s: (" + str(self.x) + "," + str(self.y) + ")"

	def __eq__(self, other):
		'''
		Summary:
			GridWorld states are equal when their x and y are the same.
		'''
		if isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y:
			return True