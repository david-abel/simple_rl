#!/usr/bin/env python

# Imports 
from simple_rl.agents import QLearnerAgent, RandomAgent
from simple_rl.tasks import TaxiOOMDP, BlockDudeOOMDP
from simple_rl.run_experiments import run_agents_on_mdp, run_single_agent_on_mdp

# Taxi initial state attributes..
agent = {"x":1, "y":1, "has_passenger":0}
passengers = [{"x":4, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}, {"x":2, "y":5, "dest_x":3, "dest_y":4, "in_taxi":0}]
walls = []
mdp = TaxiOOMDP(width=5, height=5, agent=agent, walls=walls, passengers=passengers)


ql_agent = QLearnerAgent(actions=mdp.get_actions()) 
rand_agent = RandomAgent(actions=mdp.get_actions())

viz = False
if viz == True:
	# Visualize Taxi.
	run_single_agent_on_mdp(ql_agent, mdp, episodes=50, steps=1000)
	mdp.visualize_agent(ql_agent)
else:
	# Run experiment and make plot.
	run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=15, episodes=1, steps=2000)
