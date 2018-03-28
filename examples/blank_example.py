#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import examples.srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent, DoubleQAgent
from simple_rl.tasks import FourRoomMDP, ComboLockMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Setup MDP, Agents.
    # mdp = GridWorldMDP(width=4, height=3, init_loc=(1,1), goal_locs=[(4,3)], gamma=0.95, walls=[(2,2)])
    mdp = FourRoomMDP(width=11, height=11, init_loc=(1,1), goal_locs=[(9,3)], is_goal_terminal=True, slip_prob=0.2)
    # mdp = ComboLockMDP(combo=[3,1,2], num_actions=3, num_states=3)

    dq_agent = DoubleQAgent(actions=mdp.get_actions())
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=5, episodes=1, steps=10000, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
