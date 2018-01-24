'''
Implementations of standard RL agents:

	AgentClass: Contains the basic skeleton of an RL Agent.
	QLearnerAgentClass: QLearner.
	LinearQLearnerAgentClass: Q Learner with a Linear Approximator.
	RandomAgentClass: Random actor.
	RMaxAgentClass: RMax.
	LinUCBAgentClass: Conextual Bandit Algorithm.
'''

# Grab classes.
from AgentClass import Agent
from FixedPolicyAgentClass import FixedPolicyAgent
from QLearnerAgentClass import QLearnerAgent
from DoubleQAgentClass import DoubleQAgent
from DelayedQAgentClass import DelayedQAgent
from RandomAgentClass import RandomAgent
from RMaxAgentClass import RMaxAgent

from func_approx.LinearQLearnerAgentClass import LinearQLearnerAgent
from func_approx.LinearSarsaAgentClass import LinearSarsaAgent

from bandits.LinUCBAgentClass import LinUCBAgent