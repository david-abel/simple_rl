#!/usr/bin/env python

# Python imports.
import random
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, FixedPolicyAgent
from simple_rl.tasks import RockPaperScissorsMDP
from simple_rl.run_experiments import play_markov_game 

def main(open_plot=True):
    # Setup MDP, Agents.
    markov_game = RockPaperScissorsMDP()
    ql_agent = QLearningAgent(actions=markov_game.get_actions())
    fixed_action = random.choice(markov_game.get_actions())
    fixed_agent = FixedPolicyAgent(policy=lambda s: fixed_action)

    # Run experiment and make plot.
    play_markov_game([ql_agent, fixed_agent], markov_game, instances=15, episodes=1, steps=40, open_plot=open_plot) 

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
