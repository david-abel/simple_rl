#!/usr/bin/env python

# Imports 
from simple_rl.agents import QLearnerAgent, RandomAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp 

# Setup MDP, Agents.
mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6,6)])
ql_agent = QLearnerAgent(actions=mdp.get_actions()) 
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=10, episodes=150, steps=25) 