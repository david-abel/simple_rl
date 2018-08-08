#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.utils import make_mdp
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.run_experiments import run_agents_lifelong
from simple_rl.abstraction import AbstractionWrapper, aa_helpers, ActionAbstraction

def main(open_plot=True):
    # Setup MDP, Agents.
    mdp_distr = make_mdp.make_mdp_distr(mdp_class="four_room")
    ql_agent = QLearningAgent(actions=mdp_distr.get_actions())
    rand_agent = RandomAgent(actions=mdp_distr.get_actions())

    # Make goal-based option agent.
    goal_based_options = aa_helpers.make_goal_based_options(mdp_distr)
    goal_based_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=goal_based_options)
    option_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":mdp_distr.get_actions()}, action_abstr=goal_based_aa)

    # Run experiment and make plot.
    run_agents_lifelong([ql_agent, rand_agent, option_agent], mdp_distr, samples=10, episodes=100, steps=150, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
