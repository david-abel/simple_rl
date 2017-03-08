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
from simple_rl.mdp import MarkovGameMDP

def play_markov_game(agent_dict, markov_game_mdp, num_instances=10, num_episodes=100, num_steps=30):
    '''
    Args:
        agent_dict (dict of Agents): See agents/AgentClass.py (and friends).
        markov_game_mdp (MarkovGameMDP): See mdp/markov_games/MarkovGameMDPClass.py.
        num_instances (int) [opt]: Number of times to run each agent (for confidence intervals).
        num_episodes (int) [opt]: Number of episodes for each learning instance.
        num_steps (int) [opt]: Number of times to run each agent (for confidence intervals).
    '''

    # Experiment (for reproducibility, plotting).
    exp_params = {"num_instances":num_instances} #, "num_episodes":num_episodes, "num_steps":num_steps}
    experiment = Experiment(agents=agent_dict, mdp=markov_game_mdp, params=exp_params, is_episodic=num_episodes > 1, is_markov_game=True)

    # Record how long each agent spends learning.
    print "Running experiment: \n" + str(experiment)
    start = time.clock()

    # For each instance of the agent.
    for instance in xrange(1, num_instances + 1):
        print "\tInstance " + str(instance) + " of " + str(num_instances) + "."

        reward_dict = defaultdict(str)
        action_dict = {}

        for episode in xrange(1, num_episodes + 1):
            print "\t\tEpisode " + str(episode ) + " of " + str(num_episodes) + "."
            # Compute initial state/reward.
            state = markov_game_mdp.get_init_state()

            for step in xrange(num_steps):

                # Compute each agent's policy.
                for a in agent_dict.values():
                    agent_reward = reward_dict[a.name]
                    agent_action = a.act(state, agent_reward)
                    action_dict[a.name] = agent_action

                # Terminal check.
                if state.is_terminal():
                    experiment.add_experience(agent_dict, state, action_dict, defaultdict(int), state)
                    continue

                # Execute in MDP.
                reward_dict, next_state = markov_game_mdp.execute_agent_action(action_dict)

                # Record the experience.
                experiment.add_experience(agent_dict, state, action_dict, reward_dict, next_state)

                # Update pointer.
                state = next_state

            # A final update.
            for a in agent_dict.values():
                agent_reward = reward_dict[a.name]
                agent_action = a.act(state, agent_reward)
                action_dict[a.name] = agent_action

                # Process that learning instance's info at end of learning.
                experiment.end_of_episode(a.name)

            # Reset the MDP, tell the agent the episode is over.
            markov_game_mdp.reset()

        # A final update.
        for a in agent_dict.values():
            # Reset the agent and track experiment info.
            experiment.end_of_instance(a.name)
            a.reset()

    # Time stuff.
    print "Experiment took " + str(time.clock() - start) + " seconds."

    experiment.make_plots(cumulative=True)

def run_agents_on_mdp(agents, mdp, num_instances=5, num_episodes=100, num_steps=200, clear_old_results=True, open_plot=True):
    '''
    Args:
        agents (list of Agents): See agents/AgentClass.py (and friends).
        mdp (MDP): See mdp/MDPClass.py for the abstract class. Specific MDPs in tasks/*.
        num_instances (int) [opt]: Number of times to run each agent (for confidence intervals).
        num_episodes (int) [opt]: Number of episodes for each learning instance.
        num_steps (int) [opt]: Number of steps per episode.
        clear_old_results (bool) [opt]: If true, removes all results files in the relevant results dir.

    Summary:
        Runs each agent on the given mdp according to the given parameters.
        Stores results in results/<agent_name>.csv and automatically
        generates a plot and opens it.
    '''

    # Experiment (for reproducibility, plotting).
    exp_params = {"num_instances":num_instances, "num_episodes":num_episodes, "num_steps":num_steps}
    experiment = Experiment(agents=agents, mdp=mdp, params=exp_params, is_episodic= num_episodes > 1, clear_old_results=clear_old_results)

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
                print "\t\tEpisode " + str(episode)

                # Compute initial state/reward.
                state = mdp.get_init_state()

                reward = 0
                episode_start_time = time.clock()

                for step in xrange(num_steps):

                    # Compute the agent's policy.
                    action = agent.act(state, reward)

                    # Terminal check.
                    if state.is_terminal():
                        # Self loop if in a terminal state.
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

    experiment.make_plots(open_plot=open_plot)

def choose_mdp(mdp_name, env_name="CartPole-v0"):
    '''
    Args:
        mdp_name (str): one of {gym, grid, chain, taxi, ...}
        gym_env_name (str): gym environment name, like 'CartPole-v0'

    Returns:
        (MDP)
    '''

    # Local imports.
    from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP, PrisonersDilemmaMDP, RockPaperScissorsMDP, GridGameMDP

    # Taxi MDP.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":4, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}]
    walls = []
    if mdp_name == "gym":
            from simple_rl.tasks.gym.GymMDPClass import GymMDP
            return GymMDP(env_name)
    else:
        return {"grid":GridWorldMDP(10, 10, (1, 1), (10, 10)),
                "chain":ChainMDP(15),
                "taxi":TaxiOOMDP(10, 10, slip_prob=0.0, agent_loc=agent, walls=walls, passengers=passengers),
                "random":RandomMDP(num_states=40, num_rand_trans=20),
                "prison":PrisonersDilemmaMDP(),
                "rps":RockPaperScissorsMDP(),
                "grid_game":GridGameMDP()}[mdp_name]

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp", type = str, nargs = '?', help = "Select the mdp. Options: {atari, grid, chain, taxi}")
    parser.add_argument("-env", type = str, nargs = '?', help = "Select the Atari Game or Gym environment.")
    # parser.add_argument("-debug", type = bool, nargs = '?', help = "Toggle debugging which will #print out <s,a,r,s'> during learning.}")
    args = parser.parse_args()

    # Fix variables based on options.
    task = args.mdp if args.mdp else "grid"
    env_name = args.env if args.env else "CartPole-v0"

    return task, env_name

def main():
    # Command line args.
    task, rom = parse_args()

    # Setup the MDP.
    mdp = choose_mdp(task, rom)
    actions = mdp.get_actions()
    gamma = mdp.get_gamma()

    # Setup agents.
    from simple_rl.agents import RandomAgent, QLearnerAgent
    random_agent = RandomAgent(actions)
    qlearner_agent = QLearnerAgent(actions, gamma=gamma, explore="uniform")
    
    agents = [random_agent, qlearner_agent]
    run_agents_on_mdp(agents, mdp, num_instances=3, num_episodes=1, num_steps=1000)

if __name__ == "__main__":
    main()
