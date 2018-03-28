'''
Implementations of standard RL agents:

	AgentClass: Contains the basic skeleton of an RL Agent.
	QLearningAgentClass: QLearner.
	LinearQLearningAgentClass: Q Learner with a Linear Approximator.
	RandomAgentClass: Random actor.
	RMaxAgentClass: RMax.
	LinUCBAgentClass: Conextual Bandit Algorithm.
'''

# Grab classes.
from AgentClass import Agent
from FixedPolicyAgentClass import FixedPolicyAgent
from QLearningAgentClass import QLearningAgent
from DoubleQAgentClass import DoubleQAgent
from DelayedQAgentClass import DelayedQAgent
from RandomAgentClass import RandomAgent
from RMaxAgentClass import RMaxAgent

from func_approx.LinearQAgentClass import LinearQAgent
from func_approx.LinearSarsaAgentClass import LinearSarsaAgent
try:
	from func_approx.DQNAgentClass import DQNAgent
except ImportError:
	print("Warning: Tensorflow not installed.")
	pass

from bandits.LinUCBAgentClass import LinUCBAgent