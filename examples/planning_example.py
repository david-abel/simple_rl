#!/usr/bin/env python

# Python imports.
from __future__ import print_function

# Other imports.
import srl_example_setup
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration, MCTS

def main():
    # Setup MDP, Agents.
    mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.2)

    value_iter = ValueIteration(mdp, sample_rate=5)
    mcts = MCTS(mdp, num_rollouts_per_step=50)
    # _, val = value_iter.run_vi()

    # Value Iteration.
    vi_action_seq, vi_state_seq = value_iter.plan(mdp.get_init_state())
    mcts_action_seq, mcts_state_seq = mcts.plan(mdp.get_init_state())

    print("Plan for", mdp)
    for i in range(len(mcts_action_seq)):
        print("\t", mcts_action_seq[i], mcts_state_seq[i])

if __name__ == "__main__":
    main()
