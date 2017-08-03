#!/usr/bin/env python
'''
Code for running experiments where RL agents interact with an MDP.

Instructions:
    (1) Create an MDP.
    (2) Create agents.
    (3) Set experiment parameters (instances, episodes, steps).
    (4) Call run_agents_on_mdp(agents, mdp) (or the multi_task/markov game equivalents).

    -> Runs all experiments and will open a plot with results when finished.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
import time
import argparse
import os
import math
import sys
import copy
import numpy as np
from collections import defaultdict

# Non-standard imports.
from simple_rl.experiments import Experiment
from simple_rl.mdp import MarkovGameMDP

def play_markov_game(agent_ls, markov_game_mdp, instances=10, episodes=100, steps=30):
    '''
    Args:
        agent_list (list of Agents): See agents/AgentClass.py (and friends).
        markov_game_mdp (MarkovGameMDP): See mdp/markov_games/MarkovGameMDPClass.py.
        instances (int) [opt]: Number of times to run each agent (for confidence intervals).
        episodes (int) [opt]: Number of episodes for each learning instance.
        steps (int) [opt]: Number of times to run each agent (for confidence intervals).
    '''

    # Put into dict.
    agent_dict = {}
    for a in agent_ls:
        agent_dict[a.name] = a

    # Experiment (for reproducibility, plotting).
    exp_params = {"instances":instances} #, "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agent_dict, mdp=markov_game_mdp, params=exp_params, is_episodic=episodes > 1, is_markov_game=True)

    # Record how long each agent spends learning.
    print "Running experiment: \n" + str(experiment)
    start = time.clock()

    # For each instance of the agent.
    for instance in xrange(1, instances + 1):
        print "\tInstance " + str(instance) + " of " + str(int(instances)) + "."

        reward_dict = defaultdict(str)
        action_dict = {}

        for episode in xrange(1, episodes + 1):
            sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
            sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
            sys.stdout.flush()

            # Compute initial state/reward.
            state = markov_game_mdp.get_init_state()

            for step in xrange(steps):

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
    print "Experiment took " + str(round(time.clock() - start, 2)) + " seconds."

    experiment.make_plots()

def run_agents_multi_task(agents,
                            mdp_distr,
                            task_samples=5,
                            episodes=1,
                            steps=100,
                            clear_old_results=True,
                            open_plot=True,
                            verbose=False,
                            is_rec_disc_reward=False):
    '''
    Args:
        mdp_distr
        task_samples
        episodes
        steps
    '''
    # Experiment (for reproducibility, plotting).
    exp_params = {"task_samples":task_samples, "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agents,
                mdp=mdp_distr,
                params=exp_params,
                is_episodic=episodes > 1,
                is_multi_task=True,
                clear_old_results=clear_old_results,
                is_rec_disc_reward=is_rec_disc_reward)

    # Record how long each agent spends learning.
    print "Running experiment: \n" + str(experiment)
    start = time.clock()

    times = defaultdict(float)

    # Learn.
    for agent in agents:
        print str(agent) + " is learning."
        start = time.clock()

        # --- SAMPLE NEW MDP ---
        for new_task in xrange(task_samples):
            print "  Sample " + str(new_task + 1) + " of " + str(task_samples) + "."

            # Sample the MDP.
            mdp_id = np.random.multinomial(1, mdp_distr.values()).tolist().index(1)
            mdp = mdp_distr.keys()[mdp_id]

            # Run the agent.
            run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment, verbose, is_rec_disc_reward)

            # Reset the agent.
            agent.reset()

            if "rmax" in agent.name:
                agent._reset_reward()

        # Track how much time this agent took.
        end = time.clock()
        times[agent] = round(end - start, 3)

    # Time stuff.
    print "\n--- TIMES ---"
    for agent in times.keys():
        print str(agent) + " agent took " + str(round(times[agent], 2)) + " seconds."
    print "-------------\n"

    last_reward_each_agent = defaultdict(float)
    for a in agents:
        last_reward_each_agent[a.name] = experiment.get_agent_avg_cumulative_rew(a)

    experiment.make_plots(open_plot=open_plot)

    return last_reward_each_agent

def run_agents_on_mdp(agents,
                        mdp,
                        instances=5,
                        episodes=100,
                        steps=200,
                        clear_old_results=True,
                        rew_step_count=1,
                        is_rec_disc_reward=False,
                        open_plot=True,
                        verbose=False):
    '''
    Args:
        agents (list of Agents): See agents/AgentClass.py (and friends).
        mdp (MDP): See mdp/MDPClass.py for the abstract class. Specific MDPs in tasks/*.
        instances (int) [opt]: Number of times to run each agent (for confidence intervals).
        episodes (int) [opt]: Number of episodes for each learning instance.
        steps (int) [opt]: Number of steps per episode.
        clear_old_results (bool) [opt]: If true, removes all results files in the relevant results dir.
        rew_step_count (int): Number of steps before recording reward.
        is_rec_disc_reward (bool): If true, track (and plot) discounted reward.
        open_plot (bool): If true opens the plot at the end.
        verbose (bool): If true, prints status bars per episode/instance.

    Summary:
        Runs each agent on the given mdp according to the given parameters.
        Stores results in results/<agent_name>.csv and automatically
        generates a plot and opens it.
    '''

    # Experiment (for reproducibility, plotting).
    exp_params = {"instances":instances, "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agents,
                            mdp=mdp,
                            params=exp_params,
                            is_episodic= episodes > 1,
                            clear_old_results=clear_old_results,
                            is_rec_disc_reward=is_rec_disc_reward,
                            count_r_per_n_timestep=rew_step_count)

    # Record how long each agent spends learning.
    print "Running experiment: \n" + str(experiment)
    time_dict = defaultdict(float)

    # Learn.
    for agent in agents:
        print str(agent) + " is learning."

        start = time.clock()

        # For each instance.
        for instance in xrange(1, instances + 1):
            sys.stdout.flush()
            print "  Instance " + str(instance) + " of " + str(instances) + "."
            run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment, verbose, is_rec_disc_reward)
            
            # Reset the agent.
            agent.reset()

        # print "\n"
        # Track how much time this agent took.
        end = time.clock()
        time_dict[agent] = round(end - start, 3)

    # Time stuff.
    print "\n--- TIMES ---"
    for agent in time_dict.keys():
        print str(agent) + " agent took " + str(round(time_dict[agent], 2)) + " seconds."
    print "-------------\n"

    # if not isinstance(mdp, GymMDP):
    experiment.make_plots(open_plot=open_plot)

def run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment=None, verbose=False, is_rec_disc_reward=False):
    '''
    Summary:
        Main loop of a single MDP experiment.

    Returns:
        (dict): {key=agent, val=time}
    '''

    # For each episode.
    for episode in xrange(1, episodes + 1):

        # Print episode numbers out nicely.
        sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
        sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
        sys.stdout.flush()

        # Compute initial state/reward.
        state = mdp.get_init_state()

        reward = 0
        episode_start_time = time.clock()

        # Extra printing if verbose.
        if verbose:
            print
            sys.stdout.flush()
            prog_bar_len = _make_step_progress_bar()
            _increment_bar()

        for step in xrange(steps):
            if verbose and int(prog_bar_len*float(step) / steps) > int(prog_bar_len*float(step-1) / steps):
                _increment_bar()

            # step time
            step_start = time.clock()

            # Compute the agent's policy.
            action = agent.act(state, reward)

            # Terminal check.
            if state.is_terminal():
                # Self loop if in a terminal state.
                if experiment is not None:
                    experiment.add_experience(agent, state, action, 0, state, time_taken=time.clock()-step_start)
                continue

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)

            # Record the experience.
            if experiment is not None:
                reward = mdp.get_gamma()**(step + 1) * reward if is_rec_disc_reward else reward
                experiment.add_experience(agent, state, action, reward, next_state, time_taken=time.clock()-step_start)

            # Update pointer.
            state = next_state

        # A final update.
        action = agent.act(state, reward)

        # Process experiment info at end of episode.
        if experiment is not None:
            experiment.end_of_episode(agent)

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

        if verbose:
            print "\n"

    if not verbose:
        print "\n"

    # Process that learning instance's info at end of learning.
    if experiment is not None:
        experiment.end_of_instance(agent)

def _make_step_progress_bar():
    '''
    Summary:
        Prints a step progress bar for experiments.

    Returns:
        (int): Length of the progress bar (in characters).
    '''
    progress_bar_width = 20
    sys.stdout.write("\t\t[%s]" % (" " * progress_bar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progress_bar_width+1)) # return to start of line, after '['
    return progress_bar_width

def _increment_bar():
    sys.stdout.write("-")
    sys.stdout.flush()

def choose_mdp(mdp_name, env_name="Asteroids-v0"):
    '''
    Args:
        mdp_name (str): one of {gym, grid, chain, taxi, ...}
        gym_env_name (str): gym environment name, like 'CartPole-v0'

    Returns:
        (MDP)
    '''

    # Other imports
    from simple_rl.tasks import ChainMDP, GridWorldMDP, FourRoomMDP, TaxiOOMDP, RandomMDP, PrisonersDilemmaMDP, RockPaperScissorsMDP, GridGameMDP

    # Taxi MDP.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":4, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}]
    walls = []
    if mdp_name == "gym":
            try:
                from simple_rl.tasks.gym.GymMDPClass import GymMDP
            except:
                print "Error: OpenAI gym not installed."
                quit()
            return GymMDP(env_name, render=True)
    else:
        return {"grid":GridWorldMDP(5, 5, (1, 1), goal_locs=[(5, 3), (4,1)]),
                "four_room":FourRoomMDP(),
                "chain":ChainMDP(5),
                "taxi":TaxiOOMDP(10, 10, slip_prob=0.0, agent=agent, walls=walls, passengers=passengers),
                "random":RandomMDP(num_states=40, num_rand_trans=20),
                "prison":PrisonersDilemmaMDP(),
                "rps":RockPaperScissorsMDP(),
                "grid_game":GridGameMDP(),
                "multi":{0.5:RandomMDP(num_states=40, num_rand_trans=20), 0.5:RandomMDP(num_states=40, num_rand_trans=5)}}[mdp_name]

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
    from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, LinearApproxQLearnerAgent
    
    random_agent = RandomAgent(actions)
    rmax_agent = RMaxAgent(actions, gamma=gamma, horizon=4, s_a_threshold=2)
    qlearner_agent = QLearnerAgent(actions, gamma=gamma, explore="uniform")
    lqlearner_agent = LinearApproxQLearnerAgent(actions, gamma=gamma, explore="uniform")
    agents = [qlearner_agent, random_agent]

    mdp.visualize_agent(random_agent)


    # Run Agents.
    if isinstance(mdp, MarkovGameMDP):
        # Markov Game.
        agents = {qlearner_agent.name: qlearner_agent, random_agent.name:random_agent}
        play_markov_game(agents, mdp, instances=100, episodes=1, steps=500)
    else:
        # Regular experiment.
        run_agents_on_mdp(agents, mdp, instances=50, episodes=1, steps=2000)


if __name__ == "__main__":
    main()
