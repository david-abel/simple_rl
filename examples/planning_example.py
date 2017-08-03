#!/usr/bin/env python
'''
NOTE: Incomplete. Planning infrastructure in development.
'''

# Other imports.
import srl_example_setup
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration, MCTS

# Setup MDP, Agents.
mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6,6)])
vi = ValueIteration(mdp)
mcts = MCTS(mdp)