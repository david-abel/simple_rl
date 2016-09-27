'''
Implementations of standard RL agents:

	AgentClass: Contains the basic skeleton of an RL Agent.
	QLearnerAgentClass: QLearner.
	LinearApproxQLearnerAgentClass: Q Learner with a Linear Approximator.
	RandomAgentClass: Random actor.
	RMaxAgentClass: RMax (assumes deterministic MDP for now).
'''

# Grab classes.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.QLearnerAgentClass import QLearnerAgent
from simple_rl.agents.LinearApproxQLearnerAgentClass import LinearApproxQLearnerAgent
from simple_rl.agents.RandomAgentClass import RandomAgent
from simple_rl.agents.RMaxAgentClass import RMaxAgent
from simple_rl.agents.GradientBoostingAgentClass import GradientBoostingAgent

