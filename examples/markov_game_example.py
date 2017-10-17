#!/usr/bin/env python

# Python imports.
import random

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearnerAgent, FixedPolicyAgent
from simple_rl.tasks import RockPaperScissorsMDP
from simple_rl.run_experiments import play_markov_game 

def main():
	# Setup MDP, Agents.
	markov_game = RockPaperScissorsMDP()
	ql_agent = QLearnerAgent(actions=markov_game.get_actions())
	fixed_action = random.choice(markov_game.get_actions())
	fixed_agent = FixedPolicyAgent(policy=lambda s:fixed_action)

	# Run experiment and make plot.
	play_markov_game([ql_agent, fixed_agent], markov_game, instances=15, episodes=1, steps=40) 

if __name__ == "__main__":
	main()
