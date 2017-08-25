#!/usr/bin/env python

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearnerAgent, RandomAgent, RMaxAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp 

# Setup MDP, Agents.
mdp = GridWorldMDP(width=6, height=6, init_loc=(1,1), goal_locs=[(6,6)])

rmax_agent = RMaxAgent(actions=mdp.get_actions())
ql_agent = QLearnerAgent(actions=mdp.get_actions()) 
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=10, episodes=30, steps=50, reset_at_terminal=True, include_optimal=True, clear_old_results=True)