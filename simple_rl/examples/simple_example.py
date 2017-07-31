#!/usr/bin/env python

# Imports 
from simple_rl.agents import QLearnerAgent, RMaxAgent, RandomAgent
from simple_rl.tasks import ChainMDP, GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp 

# Setup MDP, Agents.
mdp = GridWorldMDP(width=10, height=10)
ql_agent = QLearnerAgent(mdp.get_actions(), explore="uniform") 
rand_agent = RandomAgent(mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=20, episodes=100, steps=10, verbose=False) 