'''
ExperimentClass.py: Contains the Experiment Class for reproducing RL Experiments.

Purpose:
    - Stores all relevant parameters in experiment directory for easy reproducibility.
    - Auto generates plot using chart_utils.
    - Can document learning activity.
'''

# Python imports.
from __future__ import print_function
import os
from collections import defaultdict

# Other imports.
from simple_rl.utils import chart_utils
from simple_rl.experiments.ExperimentParametersClass import ExperimentParameters

class Experiment(object):

    FULL_EXP_FILE_NAME = "full_experiment.txt"
    EXP_PARAM_FILE_NAME = "exp_info.txt"

    ''' Experiment Class for RL Experiments '''

    # Dumps the results in a directory called "results" in the current working dir.
    RESULTS_DIR = os.path.join(os.getcwd(), "results", "")

    def __init__(self,
                    agents,
                    mdp,
                    agent_colors=[],
                    params=None,
                    is_episodic=False,
                    is_markov_game=False,
                    is_lifelong=False,
                    track_disc_reward=False,
                    clear_old_results=True,
                    count_r_per_n_timestep=1,
                    cumulative_plot=True,
                    exp_function="run_agents_on_mdp",
                    dir_for_plot="",
                    experiment_name_prefix=""):
        '''
        Args:
            agents (list)
            mdp (MDP)
            agent_colors (list)
            params (dict)
            is_episodic (bool)
            is_markov_game (bool)
            is_lifelong (bool)
            clear_old_results (bool)
            count_r_per_n_timestep (int)
            cumulative_plot (bool)
            exp_function (lambda): tracks with run_experiments.py function was called.
            dir_for_plot (str)
            experiment_name_prefix (str)
        '''
        # Store all relevant bools.
        self.agents = agents
        self.agent_colors = range(len(self.agents)) if agent_colors == [] else agent_colors
        params["track_disc_reward"] = track_disc_reward
        # params["is_lifelong"] = is_lifelong
        # params["agent_colors"] = agent_colors
        self.parameters = ExperimentParameters(params)
        self.mdp = mdp
        self.track_disc_reward = track_disc_reward
        self.count_r_per_n_timestep = count_r_per_n_timestep
        self.steps_since_added_r = 1
        self.rew_since_count = 0
        self.cumulative_plot = cumulative_plot
        self.name = str(self.mdp)
        self.rewards = defaultdict(list)
        self.times = defaultdict(list)
        if dir_for_plot == "":
            self.exp_directory = os.path.join(Experiment.RESULTS_DIR, self.name)
        else:
            self.exp_directory = os.path.join(os.getcwd(), dir_for_plot, self.name)

        self.experiment_name_prefix = experiment_name_prefix
        self.is_episodic = is_episodic
        self.is_markov_game = is_markov_game
        self._setup_files(clear_old_results)

        # Write experiment reproduction file.
        self._make_and_write_agent_and_mdp_params(agents, mdp, self.parameters.params, exp_function)

    def _make_and_write_agent_and_mdp_params(self, agents, mdp, parameters, exp_function):
        '''
        Args:
            agents
            mdp
            parameters

        Summary:
            Writes enough detail about @agents, @mdp, and @parameters to the file results/<exp_name>/params.txt 
            so that the function simple_rl.run_experiments.reproduce_from_exp_file can rerun the experiment.
        '''
        out_file = open(os.path.join(self.exp_directory, Experiment.FULL_EXP_FILE_NAME), "w")

        from simple_rl.mdp import OOMDP
        from simple_rl.pomdp.POMDPClass import POMDP
        from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
        if isinstance(mdp, OOMDP) or isinstance(mdp, POMDP) or isinstance(mdp, MarkovGameMDP):
            # We don't do markov games.
            return

        # MDP.
        mdp_class = str(type(mdp))
        mdp_params = mdp.get_parameters()
        out_file.write("MDP:" + mdp_class + "\n")
        for param in mdp_params:
            out_file.write("\t\t" + param + "=" + str(mdp_params[param]) + "=" + str(type(mdp_params[param])) + "\n")

        out_file.write("\n")

        # Get agents and their parameters.
        for i, agent in enumerate(agents):
            agent_params = agent.get_parameters()
            agent_class = str(i) + "-" + str(type(agent))

            out_file.write("AGENT:" + agent_class + "\n")
            for param in agent_params:
                out_file.write("\t\t" + param + "=" + str(agent_params[param]) + "=" + str(type(agent_params[param])) + "\n")
            out_file.write("\n")

        out_file.write("\n")

        # Misc. Params.
        out_file.write("MISC\n")
        for param in parameters:
            out_file.write("\t\t" + param + "=" + str(parameters[param]) + "=" + str(type(parameters[param])) + "\n")

        # Track the function called.
        out_file.write("\n\nFUNC\n\t" + exp_function + "\n")

        # Close.
        out_file.close()

    def _setup_files(self, clear_old_results=True):
        '''
        Summary:
            Creates and removes relevant directories/files.
        '''
        if not os.path.exists(os.path.join(self.exp_directory, "")):
            os.makedirs(self.exp_directory)
        elif clear_old_results:
            for agent in self.agents:
                if os.path.exists(os.path.join(self.exp_directory, str(agent)) + ".csv"):
                    os.remove(os.path.join(self.exp_directory, str(agent)) + ".csv")
        self.write_exp_info_to_file()

    def make_plots(self, open_plot=True):
        '''
        Summary:
            Makes plots for the current experiment.
        '''
        if self.is_markov_game:
            agent_name_ls = [agent_name for agent_name in self.agents.keys()]
        else:
            agent_name_ls = [a.get_name() for a in self.agents]
            
        if self.experiment_name_prefix != "":
            plot_file_name = self.experiment_name_prefix + str(self.mdp)
        else:
            plot_file_name = ""

        chart_utils.make_plots(self.exp_directory,
                                agent_name_ls,
                                episodic=self.is_episodic,
                                plot_file_name=plot_file_name,
                                cumulative=self.cumulative_plot,
                                track_disc_reward=self.track_disc_reward,
                                open_plot=open_plot)

    def _write_extra_datum_to_file(self, mdp_name, agent, datum, datum_name):
        out_file = open(os.path.join(self.exp_directory, str(agent)) + "-" + datum_name + ".csv", "a+")
        out_file.write(str(datum) + ",")
        out_file.close()

    def get_agent_avg_cumulative_rew(self, agent):
        result_file = open(os.path.join(self.exp_directory, str(agent)) + ".csv", "r")
        
        total = 0
        num_lines = 0
        for line in result_file.readlines():
            total += sum([float(datum) for datum in line.strip().split(",")[:-1]])
            num_lines += 1

        result_file.close()

        return total / num_lines

    def add_experience(self, agent, state, action, reward, next_state, time_taken=0):
        '''
        Args:
            agent (agent OR dict): if self.is_markov_game, contains a dict of agents

        Summary:
            Record any relevant information about this experience.
        '''

        # Markov Game.
        if self.is_markov_game:
            for a in agent:
                self.rewards[a] += [reward[a]]
            return

        # Regular MDP.
        if self.steps_since_added_r % self.count_r_per_n_timestep == 0:
            if self.is_markov_game and self.count_r_per_n_timestep > 1:
                raise ValueError("(simple_rl) Experiment Error: can't track markov games per step. (set rew_step_count to 1).")
            else:
                self.rewards[agent] += [self.rew_since_count + reward]
                self.times[agent] += [time_taken]
                self.rew_since_count = 0
            self.steps_since_added_r = 1
        else:
            self.rew_since_count += reward
            self.steps_since_added_r += 1

    def end_of_episode(self, agent, num_times_to_write=1):
        '''
        Args:
            agent (str)

        Summary:
            Writes reward data from this episode to file and resets the reward.
        '''
        if self.is_episodic:
            for x in range(num_times_to_write):
                self.write_datum_to_file(agent, sum(self.rewards[agent]))
                self.write_datum_to_file(agent, sum(self.times[agent]), extra_dir="times/")
        else:
            for x in range(num_times_to_write):
                for step_reward in self.rewards[agent]:
                    self.write_datum_to_file(agent, step_reward)
        self.rewards[agent] = []

    def end_of_instance(self, agent):
        '''
        Summary:
            Adds a new line to indicate we're onto a new instance.
        '''
        #
        out_file = open(os.path.join(self.exp_directory, str(agent)) + ".csv", "a+")
        out_file.write("\n")
        out_file.close()

        if os.path.isdir(os.path.join(self.exp_directory, "times", "")):
            out_file = open(os.path.join(self.exp_directory, "times", str(agent)) + ".csv", "a+")
            out_file.write("\n")
            out_file.close()

    def write_datum_to_file(self, agent, datum, extra_dir=""):
        '''
        Summary:
            Writes datum to file.
        '''
        if extra_dir != "" and not os.path.isdir(self.exp_directory + "/" + extra_dir):
            os.makedirs(os.path.join(self.exp_directory, extra_dir))
        out_file = open(os.path.join(self.exp_directory, extra_dir, str(agent)) + ".csv", "a+")
        out_file.write(str(datum) + ",")
        out_file.close()

    def write_exp_info_to_file(self):
        '''
        Summary:
            Writes relevant experiment information to a file for reproducibility.
        '''
        out_file = open(os.path.join(self.exp_directory, Experiment.EXP_PARAM_FILE_NAME), "w+")
        to_write_to_file = self._get_exp_file_string()
        out_file.write(to_write_to_file)
        out_file.close()

    def _get_exp_file_string(self):
        '''
        Returns:
            (str): contains the AGENT-names, the MDP-names, and PARAMETER-information.
        '''
        mdp_text = "(Markov Game MDP)" if self.is_markov_game else "(MDP)"
        mdp_string = mdp_text + "\n\t" + self.name + "\n"
        agent_string = "(Agents)\n"
        for i, agent in enumerate(self.agents):
            agent_string += "\t" + str(agent) + "," + str(self.agent_colors[i]) + "\n"
        param_string = "(Params)" + str(self.parameters) + "\n"

        return  mdp_string + agent_string + param_string

    def __str__(self):
        return self._get_exp_file_string()
