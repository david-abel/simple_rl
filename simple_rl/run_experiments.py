#!/usr/bin/env python
'''
Code for running experiments where RL agents interact with an MDP.

Instructions:
    (1) Set mdp in main.
    (2) Create agents.
    (3) Set experiment parameters (num_instances, num_episodes, num_steps).
    (4) Call run_agents_on_mdp(agents, mdp).

    -> Runs all experiments and will open a plot with results when finished.

Functions:
    run_agents_on_mdp: Carries out an experiment with the given agents, mdp, and parameters.
    main: Creates an MDP, a few agents, and runs an experiment.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
import time
import argparse
import os
import sys
import copy
from collections import defaultdict

from simple_rl.experiments import Experiment

def run_agents_on_mdp(agents, mdp, num_instances=1, num_episodes=2000, num_steps=200):
    '''
    Args:
        agents (list of Agents): See agents/AgentClass.py (and friends).
        mdp (MDP): See mdp/MDPClass.py for the abstract class. Specific MDPs in tasks/*.
        num_instances (int) [opt]: Number of times to run each agent (for confidence intervals).
        num_episodes (int) [opt]: Number of episodes for each learning instance.
        num_steps (int) [opt]: Number of steps per episode.

    Summary:
        Runs each agent on the given mdp according to the given parameters.
        Stores results in results/<agent_name>.csv and automatically
        generates a plot and opens it.
    '''

    # Experiment (for reproducibility, plotting).
    exp_params = {"num_instances":num_instances, "num_episodes":num_episodes, "num_steps":num_steps}
    experiment = Experiment(agents=agents, mdp=mdp, params=exp_params, is_episodic= num_episodes > 1)

    # Record how long each agent spends learning.
    times = defaultdict(float)
    print "Running experiment: \n" + str(experiment)

    # Learn.
    for agent in agents:
        print str(agent) + " is learning."
        start = time.clock()

        # For each instance of the agent.
        for instance in xrange(1, num_instances + 1):
            print "\tInstance " + str(instance) + " of " + str(num_instances) + "."

            # For each episode.
            for episode in xrange(1, num_episodes + 1):

                # Compute initial state/reward.
                state = mdp.get_init_state()

                reward = 0
                episode_start_time = time.clock()

                for step in xrange(num_steps):

                    # Compute the agent's policy.
                    action = agent.act(state, reward)

                    # Terminal check.
                    if state.is_terminal():
                        experiment.add_experience(agent, state, action, 0, state)
                        continue

                    # Execute in MDP.
                    reward, next_state = mdp.execute_agent_action(action)

                    # Record the experience.
                    experiment.add_experience(agent, state, action, reward, next_state)

                    # Update pointer.
                    state = next_state

                # A final update.
                action = agent.act(state, reward)

                # Process experiment info at end of episode.
                experiment.end_of_episode(agent)

                # Reset the MDP, tell the agent the episode is over.
                mdp.reset()
                agent.end_of_episode()

            # Process that learning instance's info at end of learning.
            experiment.end_of_instance(agent)

            # Reset the agent.
            agent.reset()

        # Track how much time this agent took.
        end = time.clock()
        times[agent] = round(end - start, 3)

    # Time stuff.
    print "\n--- TIMES ---"
    for agent in times.keys():
        print str(agent) + " agent took " + str(times[agent]) + " seconds."
    print "-------------\n"

    experiment.make_plots()

def choose_mdp(mdp_name, atari_game="centipede"):
    '''
    Args:
        mdp_name (str): one of {atari, grid, chain, taxi}
        atari_game (str): one of {centipede, breakout, etc.}

    Returns:
        (MDP)
    '''

    # Local imports.
    from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP

    # Taxi MDP.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":4, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}]
    walls = []
    if mdp_name == "atari":
        # Atari import is here in case users don't have the Arcade Learning Environment.
        # try:
        from simple_rl.tasks.atari.AtariMDPClass import AtariMDP
        return AtariMDP(rom=atari_game, grayscale=True)
        # except ImportError:
            # print "ERROR: you don't have the Arcade Learning Environment installed."
            # print "\tTry here: https://github.com/mgbellemare/Arcade-Learning-Environment."
            # quit()
    else:
        return {"grid":GridWorldMDP(10, 10, (1, 1), (10, 10)),
                "chain":ChainMDP(15),
                "taxi":TaxiOOMDP(10, 10, slip_prob=0.0, agent_loc=agent, walls=walls, passengers=passengers),
                "random":RandomMDP(num_states=40, num_rand_trans=20)}[mdp_name]

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp", type = str, nargs = '?', help = "Select the mdp. Options: {atari, grid, chain, taxi}")
    parser.add_argument("-rom", type = str, nargs = '?', help = "Select the Atari Game to run: {centipede, breakout, etc.}")
    # parser.add_argument("-debug", type = bool, nargs = '?', help = "Toggle debugging which will #print out <s,a,r,s'> during learning.}")
    args = parser.parse_args()

    # Fix variables based on options.
    task = args.mdp if args.mdp else "grid"
    atari_rom = args.rom if args.rom else "breakout"

    return task, atari_rom

def main():
    # Command line args.
    task, rom = parse_args()

    # Setup the MDP.
    mdp = choose_mdp(task, rom)
    actions = mdp.get_actions()
    gamma = mdp.get_gamma()

    # Setup agents.
    from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, LinearApproxQLearnerAgent
    random_agent = RandomAgent(actions)
    rmax_agent = RMaxAgent(actions, gamma=gamma, horizon=4)
    qlearner_agent = QLearnerAgent(actions, gamma=gamma, explore="uniform")
    softmax_qlearner_agent = QLearnerAgent(actions, gamma=gamma, explore="softmax")
    lin_approx_agent = LinearApproxQLearnerAgent(actions, gamma=gamma, explore="uniform")

    # Choose agents.    
    agents = [lin_approx_agent]
    if "task" == "atari":
        agents = [lin_approx_agent, random_agent]

    # Run experiments.
    run_agents_on_mdp(agents, mdp)

if __name__ == "__main__":
    main()
