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

# Local imports.
from ..utils import chart_utils
from ExperimentParametersClass import ExperimentParameters

class Experiment(object):
    ''' Experiment Class for RL Experiments '''

    # Dumps the results in a directory called "results" in the current working dir.
    RESULTS_DIR = os.getcwdu() + "/results/"

    def __init__(self, agents, mdp, params=None, is_episodic=False, is_markov_game=False, is_multi_task=False, clear_old_results=True):
        self.agents = agents
        self.parameters = ExperimentParameters(params)
        self.mdp = mdp
        self.is_multi_task = is_multi_task

        if self.is_multi_task:
            self.name = "multi_task"
        else:
            self.name = str(self.mdp)
            
        self.rewards = defaultdict(list)
        self.exp_directory = Experiment.RESULTS_DIR + self.name
        self.is_episodic = is_episodic
        self.is_markov_game = is_markov_game
        self._setup_files(clear_old_results)

    def _setup_files(self, clear_old_results=True):
        '''
        Summary:
            Creates and removes relevant directories/files.
        '''
        if not os.path.exists(self.exp_directory + "/"):
            os.makedirs(self.exp_directory + "/")
        elif clear_old_results:
            for agent in self.agents:
                if os.path.exists(self.exp_directory + "/" + str(agent) + ".csv"):
                    os.remove(self.exp_directory + "/" + str(agent) + ".csv")
        self.write_exp_info_to_file()

    def make_plots(self, cumulative=True, open_plot=True):
        '''
        Summary:
            Makes plots for the current experiment.
        '''
        chart_utils.make_plots(self.exp_directory, self.agents, episodic=self.is_episodic, cumulative=cumulative, open_plot=open_plot)

    def add_experience(self, agent, state, action, reward, next_state):
        '''
        Args:
            agent (agent OR dict): if self.is_markov_game, contains a dict of agents
        Summary:
            Record any relevant information about this experience.
        '''
        if self.is_markov_game:
            for a in agent:
                self.rewards[a] += [reward[a]]
        else:
            self.rewards[agent] += [reward]

    def end_of_episode(self, agent):
        '''
        Args:
            agent (str)

        Summary:
            Writes reward data from this episode to file and resets the reward.
        '''
        if self.is_episodic:
            self.write_reward_to_file(agent, sum(self.rewards[agent]))
        else:
            for step_reward in self.rewards[agent]:
                self.write_reward_to_file(agent, step_reward)
        self.rewards[agent] = []


    def end_of_instance(self, agent):
        '''
        Summary:
            Adds a new line to indicate we're onto a new instance.
        '''
        out_file = open(self.exp_directory + "/" + str(agent) + ".csv", "a+")
        out_file.write("\n")
        out_file.close()

    def write_reward_to_file(self, agent, reward):
        '''
        Summary:
            Writes reward to file.
        '''
        out_file = open(self.exp_directory + "/" + str(agent) + ".csv", "a+")
        out_file.write(str(reward) + ",")
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
