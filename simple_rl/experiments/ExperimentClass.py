'''
ExperimentClass.py: Contains the Experiment Class for reproducing RL Experiments.

Purpose:
    - Stores all relevant parameters in experiment directory for easy reproducibility.
    - Auto generates plot using chart_utils.
    - Can document learning activity.
'''

# Python imports.
import os
from collections import defaultdict

# Other imports.
from simple_rl.utils import chart_utils
from simple_rl.experiments.ExperimentParametersClass import ExperimentParameters

class Experiment(object):
    ''' Experiment Class for RL Experiments '''

    # Dumps the results in a directory called "results" in the current working dir.
    RESULTS_DIR = os.path.join(os.getcwdu(), "results", "")

    def __init__(self,
                    agents,
                    mdp,
                    params=None,
                    is_episodic=False,
                    is_markov_game=False,
                    is_multi_task=False,
                    is_rec_disc_reward=False,
                    clear_old_results=True,
                    count_r_per_n_timestep=1,
                    cumulative_plot=True):
        '''
        Args:
            agents (list)
            mdp (MDP)
            params (dict)
            is_episodic (bool)
            is_markov_game (bool)
            is_multi_task (bool)
            clear_old_results (bool)
            count_r_per_n_timestep (int)
            cumulative_plot (bool)
        '''
        self.agents = agents
        self.parameters = ExperimentParameters(params)
        self.mdp = mdp
        self.is_multi_task = is_multi_task
        self.is_rec_disc_reward = is_rec_disc_reward
        self.count_r_per_n_timestep = count_r_per_n_timestep
        self.steps_since_added_r = 1
        self.rew_since_count = 0
        self.cumulative_plot = cumulative_plot

        if self.is_multi_task:
            self.name = "multitask-" + str(self.mdp.keys()[0])
        else:
            self.name = str(self.mdp)
            
        self.rewards = defaultdict(list)
        self.times = defaultdict(list)
        self.exp_directory = Experiment.RESULTS_DIR + self.name
        self.is_episodic = is_episodic
        self.is_markov_game = is_markov_game
        self._setup_files(clear_old_results)

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
        chart_utils.make_plots(self.exp_directory,
                                self.agents,
                                episodic=self.is_episodic,
                                cumulative=self.cumulative_plot,
                                is_rec_disc_reward=self.is_rec_disc_reward,
                                open_plot=open_plot)

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
        if self.steps_since_added_r % self.count_r_per_n_timestep == 0:
            if self.is_markov_game:
                for a in agent:
                    self.rewards[a] += [reward[a]]
            else:
                self.rewards[agent] += [reward]
                self.times[agent] += [time_taken]
            self.steps_since_added_r = 1
        else:
            if self.is_markov_game:
                for a in agent:
                    self.rew_since_count[a] += [reward[a]]
            else:
                self.rew_since_count += reward
            self.steps_since_added_r += 1

    def end_of_episode(self, agent):
        '''
        Args:
            agent (str)

        Summary:
            Writes reward data from this episode to file and resets the reward.
        '''
        if self.is_episodic:
            self.write_datum_to_file(agent, sum(self.rewards[agent]))
            self.write_datum_to_file(agent, sum(self.times[agent]), extra_dir="times/")
        else:
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
        out_file = open(self.exp_directory + "/parameters.txt", "w+")
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
        for agent in self.agents:
            agent_string += "\t" + str(agent) + "\n"
        param_string = "(Params)" + str(self.parameters) + "\n"

        return  mdp_string + agent_string + param_string

    def __str__(self):
        return self._get_exp_file_string()
