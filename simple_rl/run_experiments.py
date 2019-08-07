#!/usr/bin/env python
'''
Code for running experiments where RL agents interact with an MDP.

Instructions:
    (1) Create an MDP.
    (2) Create agents.
    (3) Set experiment parameters (instances, episodes, steps).
    (4) Call run_agents_on_mdp(agents, mdp) (or the lifelong/markov game equivalents).

    -> Runs all experiments and will open a plot with results when finished.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
from __future__ import print_function
import time
import argparse
import os
import math
import sys
import copy
import numpy as np
from collections import defaultdict

# Non-standard imports.
from simple_rl.planning import ValueIteration
from simple_rl.experiments import Experiment
from simple_rl.mdp import MarkovGameMDP
from simple_rl.utils import chart_utils
from simple_rl.agents import *
from simple_rl.tasks import *


def play_markov_game(agent_ls, markov_game_mdp, instances=10, episodes=100, steps=30, verbose=False, open_plot=True):
    '''
    Args:
        agent_list (list of Agents): See agents/AgentClass.py (and friends).
        markov_game_mdp (MarkovGameMDP): See mdp/markov_games/MarkovGameMDPClass.py.
        instances (int): Number of times to run each agent (for confidence intervals).
        episodes (int): Number of episodes for each learning instance.
        steps (int): Number of times to run each agent (for confidence intervals).
        verbose (bool)
        open_plot (bool): If true opens plot.
    '''

    # Put into dict.
    agent_dict = {}
    for a in agent_ls:
        agent_dict[a.name] = a

    # Experiment (for reproducibility, plotting).
    exp_params = {"instances":instances, "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agent_dict, mdp=markov_game_mdp, params=exp_params, is_episodic=episodes > 1, is_markov_game=True)

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    start = time.clock()

    # For each instance of the agent.
    for instance in range(1, instances + 1):
        print("\tInstance " + str(instance) + " of " + str(int(instances)) + ".")

        reward_dict = defaultdict(str)
        action_dict = {}

        for episode in range(1, episodes + 1):
            if verbose:
                sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
                sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
                sys.stdout.flush()

            # Compute initial state/reward.
            state = markov_game_mdp.get_init_state()

            for step in range(steps):

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
    print("Experiment took " + str(round(time.clock() - start, 2)) + " seconds.")

    experiment.make_plots(open_plot=open_plot)

def run_agents_lifelong(agents,
                            mdp_distr,
                            samples=5,
                            episodes=1,
                            steps=100,
                            clear_old_results=True,
                            open_plot=True,
                            verbose=False,
                            track_disc_reward=False,
                            reset_at_terminal=False,
                            resample_at_terminal=False,
                            cumulative_plot=True,
                            dir_for_plot="results"):
    '''
    Args:
        agents (list)
        mdp_distr (MDPDistribution)
        samples (int)
        episodes (int)
        steps (int)
        clear_old_results (bool)
        open_plot (bool)
        verbose (bool)
        track_disc_reward (bool): If true records and plots discounted reward, discounted over episodes. So, if
            each episode is 100 steps, then episode 2 will start discounting as though it's step 101.
        reset_at_terminal (bool)
        resample_at_terminal (bool)
        cumulative_plot (bool)
        dir_for_plot (str)

    Summary:
        Runs each agent on the MDP distribution according to the given parameters.
        If @mdp_distr has a non-zero horizon, then gamma is set to 1 and @steps is ignored.
    '''
    # Experiment (for reproducibility, plotting).
    exp_params = {"samples":samples, "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agents,
                    mdp=mdp_distr,
                    params=exp_params,
                    is_episodic=episodes > 1,
                    is_lifelong=True,
                    clear_old_results=clear_old_results,
                    track_disc_reward=track_disc_reward,
                    cumulative_plot=cumulative_plot,
                    dir_for_plot=dir_for_plot)

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    start = time.clock()

    times = defaultdict(float)

    # Learn.
    for agent in agents:
        print(str(agent) + " is learning.")
        start = time.clock()

        # --- SAMPLE NEW MDP ---
        for new_task in range(samples):
            print("  Sample " + str(new_task + 1) + " of " + str(samples) + ".")

            # Sample the MDP.
            mdp = mdp_distr.sample()

            # Run the agent.
            hit_terminal, total_steps_taken, _ = run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment, verbose, track_disc_reward, reset_at_terminal, resample_at_terminal)

            # If we resample at terminal, keep grabbing MDPs until we're done.
            while resample_at_terminal and hit_terminal and total_steps_taken < steps:
                mdp = mdp_distr.sample()
                hit_terminal, steps_taken, _ = run_single_agent_on_mdp(agent, mdp, episodes, steps - total_steps_taken, experiment, verbose, track_disc_reward, reset_at_terminal, resample_at_terminal)
                total_steps_taken += steps_taken

            # Reset the agent.
            agent.reset()

        # Track how much time this agent took.
        end = time.clock()
        times[agent] = round(end - start, 3)


    # Time stuff.
    print("\n--- TIMES ---")
    for agent in times.keys():
        print(str(agent) + " agent took " + str(round(times[agent], 2)) + " seconds.")
    print("-------------\n")

    experiment.make_plots(open_plot=open_plot)

def run_agents_on_mdp(agents,
                        mdp,
                        instances=5,
                        episodes=100,
                        steps=200,
                        clear_old_results=True,
                        rew_step_count=1,
                        track_disc_reward=False,
                        open_plot=True,
                        verbose=False,
                        reset_at_terminal=False,
                        cumulative_plot=True,
                        dir_for_plot="results",
                        experiment_name_prefix="",
                        track_success=False,
                        success_reward=None):
    '''
    Args:
        agents (list of Agents): See agents/AgentClass.py (and friends).
        mdp (MDP): See mdp/MDPClass.py for the abstract class. Specific MDPs in tasks/*.
        instances (int): Number of times to run each agent (for confidence intervals).
        episodes (int): Number of episodes for each learning instance.
        steps (int): Number of steps per episode.
        clear_old_results (bool): If true, removes all results files in the relevant results dir.
        rew_step_count (int): Number of steps before recording reward.
        track_disc_reward (bool): If true, track (and plot) discounted reward.
        open_plot (bool): If true opens the plot at the end.
        verbose (bool): If true, prints status bars per episode/instance.
        reset_at_terminal (bool): If true sends the agent to the start state after terminal.
        cumulative_plot (bool): If true makes a cumulative plot, otherwise plots avg. reward per timestep.
        dir_for_plot (str): Path
        experiment_name_prefix (str): Adds this to the end of the usual experiment name.
        track_success (bool): If true, tracks whether each run is successful and generates an additional success plot at the end.
        success_reward (int): If set, determines the success criteria.

    Summary:
        Runs each agent on the given mdp according to the given parameters.
        Stores results in results/<agent_name>.csv and automatically
        generates a plot and opens it.
    '''
    if track_success and success_reward is None:
        raise ValueError("(simple_rl): run_agents_on_mdp must set param @success_reward when @track_success=True.")

    # Experiment (for reproducibility, plotting).
    exp_params = {"instances":instances, "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agents,
                            mdp=mdp,
                            params=exp_params,
                            is_episodic= episodes > 1,
                            clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward,
                            count_r_per_n_timestep=rew_step_count,
                            cumulative_plot=cumulative_plot,
                            dir_for_plot=dir_for_plot,
                            experiment_name_prefix=experiment_name_prefix,
                            track_success=track_success,
                            success_reward=success_reward)

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    time_dict = defaultdict(float)

    # Learn.
    for agent in agents:
        print(str(agent) + " is learning.")

        start = time.clock()

        # For each instance.
        for instance in range(1, instances + 1):
            print("  Instance " + str(instance) + " of " + str(instances) + ".")
            sys.stdout.flush()
            run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment, verbose, track_disc_reward, reset_at_terminal=reset_at_terminal)
            if "fixed" in agent.name:
                break

            # Reset the agent.
            agent.reset()
            mdp.end_of_instance()
        # Track how much time this agent took.
        end = time.clock()
        time_dict[agent] = round(end - start, 3)
        print()
        

    # Time stuff.
    print("\n--- TIMES ---")
    for agent in time_dict.keys():
        print(str(agent) + " agent took " + str(round(time_dict[agent], 2)) + " seconds.")
    print("-------------\n")

    experiment.make_plots(open_plot=open_plot)

def run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment=None, verbose=False, track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False):
    '''
    Summary:
        Main loop of a single MDP experiment.

    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, list: cumulative discounted reward per episode)
    '''
    if reset_at_terminal and resample_at_terminal:
        raise ValueError("(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")

    value_per_episode = [0] * episodes
    gamma = mdp.get_gamma()

    # For each episode.
    for episode in range(1, episodes + 1):

        cumulative_episodic_reward = 0

        if verbose:
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
            print()
            sys.stdout.flush()
            prog_bar_len = _make_step_progress_bar()

        for step in range(1, steps + 1):
            if verbose and int(prog_bar_len*float(step) / steps) > int(prog_bar_len*float(step-1) / steps):
                _increment_bar()

            # step time
            step_start = time.clock()

            # Compute the agent's policy.
            action = agent.act(state, reward)

            # Terminal check.
            if state.is_terminal():

                if verbose:
                    sys.stdout.write("x")

                if episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                    # Self loop if we're not episodic or resetting and in a terminal state.
                    experiment.add_experience(agent, state, action, 0, state, time_taken=time.clock()-step_start)
                    continue
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)

            # Track value.
            value_per_episode[episode - 1] += reward * gamma ** step
            cumulative_episodic_reward += reward

            # Record the experience.
            if experiment is not None:
                reward_to_track = mdp.get_gamma()**(step + 1 + episode*steps) * reward if track_disc_reward else reward
                reward_to_track = round(reward_to_track, 5)
                experiment.add_experience(agent, state, action, reward_to_track, next_state, time_taken=time.clock() - step_start)

            if next_state.is_terminal():
                if reset_at_terminal:
                    # Reset the MDP.
                    next_state = mdp.get_init_state()
                    mdp.reset()
                elif resample_at_terminal and step < steps:
                    mdp.reset()
                    return True, step, value_per_episode

            # Update pointer.
            state = next_state

        # Process experiment info at end of episode.
        if experiment is not None:
            experiment.end_of_episode(agent)
            print

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

        if verbose:
            print("\n")

    # Process that learning instance's info at end of learning.
    if experiment is not None:
        experiment.end_of_instance(agent)

    # Only print if our experiment isn't trivially short.
    if verbose:
        print("\tLast episode reward:", cumulative_episodic_reward)

    return False, steps, value_per_episode

def run_single_belief_agent_on_pomdp(belief_agent, pomdp, episodes, steps, experiment=None, verbose=False,
                                     track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False):
    '''

    Args:
        belief_agent:
        pomdp:
        episodes:
        steps:
        experiment:
        verbose:
        track_disc_reward:
        reset_at_terminal:
        resample_at_terminal:

    Returns:

    '''
    pass

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

def evaluate_agent(agent, mdp, instances=10, episodes=1, steps=None):
    '''
    Args:
        agent (simple_rl.Agent)
        mdp (simple_rl.MDP)
        instances (int)
        episodes (int)
        steps (int)

    Returns:
        (float): Avg. cumulative discounted reward.
    '''
    total = 0.0
    steps = int(1 / (1 - mdp.get_gamma())) * 10 if steps is None else steps
    for i in range(instances):
        _, _, val_per_episode = run_single_agent_on_mdp(agent, mdp, episodes=episodes, steps=steps)
        total += val_per_episode[-1]

        # Reset the agent.
        agent.reset()
        mdp.reset()
        mdp.end_of_instance()

    return total / instances

def reproduce_from_exp_file(exp_name, results_dir="results", open_plot=True):
    '''
    Args:
        exp_name (str)
        results_dir (str)
        open_plot (bool)

    Summary:
        Extracts the agents, MDP, and parameters from the file and runs the experiment.
        Stores data in "results_dir/exp_name/reproduce_i/*", where "i" is determined
        based on the existence of earlier "reproduce" files.
    '''

    # Get dir and file.
    exp_dir = os.path.join(results_dir, exp_name)
    exp_file = Experiment.FULL_EXP_FILE_NAME
    full_exp_file = os.path.join(exp_dir, exp_file)

    # Check to make sure the file exists.
    if not os.path.exists(full_exp_file):
        raise ImportError("(simple_rl): no such experiment: " + str(full_exp_file) + ".")

    # Open the file.
    import json
    from simple_rl.utils.additional_datastructures import TupleEncoder
    json_data_file = open(full_exp_file, "r")
    all_exp_info = json.load(json_data_file, object_hook=TupleEncoder.hinted_tuple_hook)
    json_data_file.close()

    # Placeholders.
    experiment_param_dict = {}
    actions = []
    experiment_func = None

    # Make MDP.
    full_mdp_class_str = all_exp_info["MDP"]["name"]
    mdp_class_str = full_mdp_class_str[full_mdp_class_str.find("'") + 1 : full_mdp_class_str.rfind("'")].split(".")[-1]
    mdp_param_dict = all_exp_info["MDP"]["params"]
    MDPClass = globals()[mdp_class_str]
    mdp = MDPClass(**mdp_param_dict)

    # Make Agents.
    agents = []
    for full_agent_class_str in all_exp_info["AGENTS"].keys():
        # Convert full str into class name.
        agent_class_str = full_agent_class_str[full_agent_class_str.find("'") + 1 : full_agent_class_str.rfind("'")].split(".")[-1]

        # Get class and make agent.
        agent_param_dict = all_exp_info["AGENTS"][full_agent_class_str]["params"]
        agent_index = all_exp_info["AGENTS"][full_agent_class_str]["index"]
        AgentClass = globals()[agent_class_str]

        # Create agent.
        agent_param_dict["actions"] = mdp.get_actions()
        agent = AgentClass(**agent_param_dict)
        agents.insert(agent_index, agent)

    # Experiment parameters.
    experiment_param_dict = all_exp_info["MISC"]

    # Function.
    experiment_func = eval(all_exp_info["FUNC"])

    # Prints.
    print("\n" + "%"*17)
    print("%"*2, "Reproducing", "%"*2)
    print("%"*17, "\n")
    print("MDP:", "\n  " + str(mdp) + "\n")
    print("Agents:")
    for a in agents:
        print("  ", a)
    print("\n" + "%"*17)
    print("%"*17, "\n")

    # Reproduce.
    chart_utils.CUSTOM_TITLE = "Reproduction: " + chart_utils._format_title(str(mdp))
    experiment_func(agents, mdp, dir_for_plot=results_dir, experiment_name_prefix="reproduce_", open_plot=open_plot, **experiment_param_dict)

    print("\n" + "%"*22)
    print("%"*2, "Done Reproducing", "%"*2)
    print("%"*22, "\n")

def _get_params_from_lines(lines, start_index):
    '''
    Args:
        lines (list)
        start_index (int)

    Returns:
        (dict)

    Summary:

    '''
    from ast import literal_eval as make_tuple
    import ast
    agent_param_dict = {}

    # Get the class.

    hit_next_agent = False
    i = start_index
    while not hit_next_agent:
        if len(lines[i]) <= 1:
            break

        # Grab param name, value, and type.
        next_line = [item.strip() for item in lines[i].split("=")]
        param_name, param_val, param_type = next_line[0], next_line[1], next_line[2][next_line[2].find("'") + 1 : next_line[2].rfind("'")]
        
        if param_type == "bool":
            param_val = bool(param_val == "True")
        elif param_type == "tuple":
            param_val = make_tuple(param_val)
        elif param_type == "list":
            param_val = ast.literal_eval(param_val)
        else:         
            param_val = eval(param_type)(param_val)
        agent_param_dict[param_name] = param_val

        i += 1

    return agent_param_dict


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
        # OpenAI Gym MDP.
        try:
            from simple_rl.tasks.gym.GymMDPClass import GymMDP
        except:
            raise ValueError("(simple_rl) Error: OpenAI gym not installed.")
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
    parser.add_argument("-env", type = str, nargs = '?', help = "Select the Gym environment.")
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
    from simple_rl.agents import RandomAgent, QLearningAgent
    
    random_agent = RandomAgent(actions)
    qlearner_agent = QLearningAgent(actions, gamma=gamma, explore="uniform")
    agents = [qlearner_agent, random_agent]

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
