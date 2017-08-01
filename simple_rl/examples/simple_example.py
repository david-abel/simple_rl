#!/usr/bin/env python

# Imports 
from simple_rl.agents import QLearnerAgent, RandomAgent
from simple_rl.tasks import ChainMDP, GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp 

# Setup MDP, Agents.
mdp = GridWorldMDP(width=8, height=8, goal_locs=[(8,8)])
ql_agent = QLearnerAgent(actions=mdp.get_actions()) 
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=15, episodes=200, steps=40) 