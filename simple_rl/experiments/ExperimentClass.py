'''
ExperimentClass.py: Contains the Experiment Class for reproducing RL Experiments.

Purpose:
    - Stores all relevant parameters in experiment directory for easy reproducibility.
    - Auto generates plot using chart_utils.
    - Can document learning activity.
'''

# Python libs.
import os
from collections import defaultdict

# Local libs.
from simple_rl.utils import chart_utils
from simple_rl.experiments.ExperimentParametersClass import ExperimentParameters

class Experiment(object):
    ''' Experiment Class for RL Experiments '''

    # Dumps the results in a directory called "results" in the current working dir.
    RESULTS_DIR = os.getcwdu() + "/results/"

    def __init__(self, agents, mdp, params=None):
        self.agents = agents
        self.parameters = ExperimentParameters(params)
        self.mdp = mdp
        self.rewards = defaultdict(list)
        self.name = str(self.mdp)
        self.exp_directory = Experiment.RESULTS_DIR + self.name
        self._setup_files()

    def _setup_files(self):
        '''
        Summary:
            Creates and removes relevant directories/files.
        '''
        if not os.path.exists(self.exp_directory + "/"):
            os.makedirs(self.exp_directory + "/")
        else:
            for agent in self.agents:
                if os.path.exists(self.exp_directory + "/" + str(agent) + ".csv"):
                    os.remove(self.exp_directory + "/" + str(agent) + ".csv")
        self.write_exp_info_to_file()

    def make_plots(self):
        '''
        Summary:
            Makes plots for the current experiment.
        '''
        chart_utils.make_plots(self.exp_directory, self.agents)

    def add_experience(self, agent, state, action, reward, next_state):
        '''
        Summary:
            Record any relevant information about this experience.
        '''
        self.rewards[agent] += [reward]

    def end_of_episode(self, agent):
        '''
        Summary:
            Writes episode data to file and resets the reward.
        '''
        self.write_episode_reward_to_file(agent, sum(self.rewards[agent]))
        self.rewards[agent] = []

    def end_of_instance(self, agent):
        '''
        Summary:
            Adds a new line to indicate we're onto a new instance.
        '''
        out_file = open(self.exp_directory + "/" + str(agent) + ".csv", "a+")
        out_file.write("\n")
        out_file.close()

    def write_episode_reward_to_file(self, agent, reward):
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
        mdp_string = "(MDP)\n\t" + str(self.mdp) + "\n"
        agent_string = "(Agents)\n"
        for agent in self.agents:
            agent_string += "\t" + str(agent) + "\n"
        param_string = "(Params)" + str(self.parameters) + "\n"

        return  mdp_string + agent_string + param_string

    def __str__(self):
        return self._get_exp_file_string()
