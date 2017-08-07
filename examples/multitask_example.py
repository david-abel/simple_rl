#!/usr/bin/env python

# Python imports.
import random as rnd

# Other imports
import srl_example_setup
from simple_rl.mdp import MDPDistribution
from simple_rl.tasks import GridWorldMDP, RandomMDP, ChainMDP, TaxiOOMDP, FourRoomMDP
from simple_rl.agents import QLearnerAgent, RandomAgent, RMaxAgent
from simple_rl.run_experiments import run_agents_multi_task

def make_mdp_distr(mdp_class="grid", num_mdps=15, gamma=0.99):
    '''
    Args:
        mdp_class (str): one of {"grid", "random"}
        num_mdps (int)

    Returns:
        (MDPDistribution)
    '''
    mdp_dist_dict = {}
    mdp_prob = 1.0 / num_mdps
    height, width = 10, 10

    # Make @num_mdps MDPs.
    for i in xrange(num_mdps):
        next_goals = rnd.sample([(1,7),(7,1),(7,7),(6,6),(6,1),(1,6)], 2)
        new_mdp = {"grid":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=rnd.sample(zip(range(1, width+1),[height]*width), 1), is_goal_terminal=True, gamma=gamma),
                    "four_room":FourRoomMDP(width=8, height=8, goal_locs=next_goals, gamma=gamma),
                    "chain":ChainMDP(num_states=10, reset_val=rnd.choice([0, 0.01, 0.05, 0.1]), gamma=gamma),
                    "random":RandomMDP(num_states=40, num_rand_trans=rnd.randint(1,10), gamma=gamma)}[mdp_class]

        mdp_dist_dict[new_mdp] = mdp_prob

    return MDPDistribution(mdp_dist_dict)


def main():
    # Make MDP distribution, agents.
    mdp_distr = make_mdp_distr(mdp_class="grid")
    ql_agent = QLearnerAgent(actions=mdp_distr.get_actions())
    rand_agent = RandomAgent(actions=mdp_distr.get_actions())

    # Run experiment and make plot.
    run_agents_multi_task([ql_agent, rand_agent], mdp_distr, task_samples=30, episodes=100, steps=50, reset_at_terminal=True)

if __name__ == "__main__":
    main()