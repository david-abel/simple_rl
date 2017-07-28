#!/usr/bin/env python

# Imports 
from simple_rl.agents import QLearnerAgent, RMaxAgent, RandomAgent
from simple_rl.tasks import ChainMDP, GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp 

# Setup MDP, Agents.
mdp = ChainMDP(5)
ql_agent = QLearnerAgent(mdp.get_actions()) 
rm_agent = RMaxAgent(mdp.get_actions()) 
rand_agent = RandomAgent(mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([ql_agent, rm_agent], mdp, instances=20, episodes=10, steps=200, verbose=False) 