#!/usr/bin/env python

# Python imports.
from __future__ import print_function

# Other imports.
import srl_example_setup
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration

def main():
    # Setup MDP, Agents.
    mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.2)
    value_iter = ValueIteration(mdp, sample_rate=5)
    value_iter.run_vi()

    # Value Iteration.
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    print("Plan for", mdp)
    for i in range(len(action_seq)):
        print("\t", action_seq[i], state_seq[i])

if __name__ == "__main__":
    main()
