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
from LinearApproxQLearnerAgentClass import LinearApproxQLearnerAgent
from RandomAgentClass import RandomAgent
from RMaxAgentClass import RMaxAgent

# Only grab the Gradient Booster if sklearn is installed.
import sys
if "sklearn" in sys.modules:
	from GradientBoostingAgentClass import GradientBoostingAgent