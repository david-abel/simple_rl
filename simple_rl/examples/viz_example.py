#!/usr/bin/env python

# Imports 
from simple_rl.agents import QLearnerAgent
from simple_rl.run_experiments import run_single_agent_on_mdp 
from simple_rl.tasks import FourRoomMDP

# Setup MDP, Agents.
dim = 9
mdp = FourRoomMDP(dim, dim, goal_locs=[(dim,dim)])
ql_agent = QLearnerAgent(mdp.get_actions()) 


run_single_agent_on_mdp(ql_agent, mdp, episodes=500, steps=200)

s = mdp.get_init_state()

# Run experiment and make plot.
mdp.visualize_agent(ql_agent)
# mdp.visualize_value(ql_agent)