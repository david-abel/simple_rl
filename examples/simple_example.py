#!/usr/bin/env python

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearnerAgent, RandomAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp 

# Setup MDP, Agents.
mdp = GridWorldMDP(width=10, height=10, init_loc=(1,1), goal_locs=[(10,10)])
ql_agent = QLearnerAgent(actions=mdp.get_actions()) 
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=5, episodes=100, steps=150)
