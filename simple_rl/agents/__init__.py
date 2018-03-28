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
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.FixedPolicyAgentClass import FixedPolicyAgent
from simple_rl.agents.QLearningAgentClass import QLearningAgent
from simple_rl.agents.DoubleQAgentClass import DoubleQAgent
from simple_rl.agents.DelayedQAgentClass import DelayedQAgent
from simple_rl.agents.RandomAgentClass import RandomAgent
from simple_rl.agents.RMaxAgentClass import RMaxAgent

from simple_rl.agents.func_approx.LinearQAgentClass import LinearQAgent
from simple_rl.agents.func_approx.LinearSarsaAgentClass import LinearSarsaAgent
try:
	from simple_rl.agents.func_approx.DQNAgentClass import DQNAgent
except ImportError:
	print("Warning: Tensorflow not installed.")
	pass

from simple_rl.agents.bandits.LinUCBAgentClass import LinUCBAgent