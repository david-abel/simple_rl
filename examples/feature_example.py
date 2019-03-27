#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent, LinearQAgent
from simple_rl.tasks import GridWorldMDP, PuddleMDP
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.abstraction import FeatureWrapper, TileCoding, BucketCoding, RBFCoding

def main(open_plot=True):
    # Setup MDP.
    mdp = PuddleMDP()

    # Make feature mappers.
    tile_coder = TileCoding(ranges=[[0, 1.0], [0, 1.0]], num_tiles=[4, 5], num_tilings=4)
    bucket_coder = BucketCoding(feature_max_vals=[1.0, 1.0], num_buckets=5)
    rbf_coder = RBFCoding()
    
    # Make agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())

    # Tabular agent w/ features.
    tile_coding_agent = FeatureWrapper(QLearningAgent, feature_mapper=tile_coder, agent_params={"actions":mdp.get_actions()})
    bucket_coding_agent = FeatureWrapper(QLearningAgent, feature_mapper=bucket_coder, agent_params={"actions":mdp.get_actions()})

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, bucket_coding_agent], mdp, instances=10, episodes=100, steps=150, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
