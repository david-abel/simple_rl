'''
Implementations of standard RL agents:

	AgentClass: Contains the basic skeleton of an RL Agent.
	QLearnerAgentClass: QLearner.
	LinearApproxQLearnerAgentClass: Q Learner with a Linear Approximator.
	RandomAgentClass: Random actor.
	RMaxAgentClass: RMax (assumes deterministic MDP for now).
'''

# Grab classes.
from AgentClass import Agent
from FixedPolicyAgentClass import FixedPolicyAgent
from QLearnerAgentClass import QLearnerAgent
from RandomAgentClass import RandomAgent
from RMaxAgentClass import RMaxAgent
from MCTSAgentClass import MCTSAgent

from func_approx.LinearApproxQLearnerAgentClass import LinearApproxQLearnerAgent
from func_approx.LinearApproxSarsaAgentClass import LinearApproxSarsaAgent
from bandits.LinUCBAgentClass import LinUCBAgent