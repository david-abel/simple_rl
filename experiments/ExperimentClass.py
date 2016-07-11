# Python libs.
import os
from collections import defaultdict

# Local libs.
from ExperimentParametersClass import ExperimentParameters

class Experiment(object):
    ''' Experiment Class for Discrete MDP Experiments '''

    resultsDir = "../results/"

    def __init__(self, agents, mdp, params=None):
        self.agents = agents
        self.parameters = ExperimentParameters(params)
        self.mdp = mdp
        self.rewards = defaultdict(list)

    def makePlots(self):
        self.writeToFile()

    def addExperience(self, agent, s, a, r, sprime):
        self.rewards[agent] += [r]

    def writeToFile(self):
        '''
        Summary:
            Writes relevant experiment information to a file for reproducibility.
        '''
        # Make the subdirectory if it doesn't yet exist.
        new_dir = Experiment.resultsDir + str(self.mdp) + "/"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        
        # Write.
        out_file = open(new_dir + "parameters.txt", "w+")
        toFile = self._getExpFileString()
        out_file.write(toFile)
        out_file.close()

    def _getExpFileString(self):
        '''
        Returns:
            (str): contains the AGENT-names, the MDP-names, and PARAMETER-information.
        '''
        agentString = "AGENTS: "
        for agent in self.agents:
            agentString += str(agent)

        agentString += "\n\n"
        mdpString = "MDP: " + str(self.mdp) + "\n\n"
        paramString = "PARAMS: " + str(self.parameters) + "\n\n"

        return agentString + mdpString + paramString















