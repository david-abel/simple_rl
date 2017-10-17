#!/usr/bin/env python

# Other imports.
import srl_example_setup
from simple_rl.agents import LinearQLearnerAgent, RandomAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main():
	# Gym MDP
	gym_mdp = GymMDP(env_name='CartPole-v0', render=False)
	num_feats = gym_mdp.get_num_state_feats()

	# Setup agents and run.
	lin_agent = LinearQLearnerAgent(gym_mdp.actions, num_features=num_feats, alpha=0.4, epsilon=0.4, anneal=True)
	rand_agent = RandomAgent(gym_mdp.actions)
	run_agents_on_mdp([lin_agent, rand_agent], gym_mdp, instances=10, episodes=30, steps=10000)

if __name__ == "__main__":
	main()
