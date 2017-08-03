#!/usr/bin/env python

# Other imports.
import srl_example_setup
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration, MCTS

# Setup MDP, Agents.
mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6,6)])
vi = ValueIteration(mdp)
mcts = MCTS(mdp)

# Run experiment and make plot.
# run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=10, episodes=150, steps=25) 