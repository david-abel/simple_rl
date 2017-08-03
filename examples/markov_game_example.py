#!/usr/bin/env python

# Imports 
from simple_rl.agents import QLearnerAgent, RandomAgent
from simple_rl.tasks import GridGameMDP, RockPaperScissorsMDP
from simple_rl.run_experiments import play_markov_game 

# Setup MDP, Agents.
mdp = GridGameMDP()
ql_agent = QLearnerAgent(actions=mdp.get_actions()) 
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
play_markov_game([ql_agent, rand_agent], mdp, instances=15, episodes=200, steps=40) 