# Python libs.
import os
import sys
from collections import defaultdict

# Local libs.
from simpleRL.utils import chartUtils
from simpleRL.experiments.ExperimentParametersClass import ExperimentParameters

class Experiment(object):
    ''' Experiment Class for Discrete MDP Experiments '''

    # Dumps the results in a directory called "results" in the current working dir.
    resultsDir = os.getcwdu() + "/results/"

    def __init__(self, agents, mdp, params=None):
        self.agents = agents
        self.parameters = ExperimentParameters(params)
        self.mdp = mdp
        self.rewards = defaultdict(list)
        self.name = str(self.mdp)
        self.expDirectory = Experiment.resultsDir + self.name
        self._setupFiles()

    def _setupFiles(self):
        '''
        Summary:
            Creates and removes relevant directories/files.
        '''
        if not os.path.exists(self.expDirectory + "/"):
            os.makedirs(self.expDirectory + "/")
        else:
            for agent in self.agents:
                if os.path.exists(self.expDirectory + "/" + str(agent) + ".csv"):
                    os.remove(self.expDirectory + "/" + str(agent) + ".csv")

    def makePlots(self):
        chartUtils.makePlots(self.expDirectory, self.agents)

    def addExperience(self, agent, s, a, r, sprime):
        self.rewards[agent] += [r]

    def endOfEpisode(self, agent):
        self.writeEpisodeRewardToFile(agent, sum(self.rewards[agent]))
        self.rewards[agent] = []

    def endOfInstance(self, agent):
        '''
        Summary:
            Adds a new line to indicate we're onto a new instance.
        '''
        outFile = open(self.expDirectory + "/" + str(agent) + ".csv", "a+")
        outFile.write("\n")
        outFile.close()

    def writeEpisodeRewardToFile(self, agent, r):
        # Write reward.
        outFile = open(self.expDirectory + "/" + str(agent) + ".csv", "a+")
        outFile.write(str(r) + ",")
        outFile.close()

    def writeToFile(self):
        '''
        Summary:
            Writes relevant experiment information to a file for reproducibility.
        '''
        outFile = open(self.expDirectory + "/parameters.txt", "w+")
        toFile = self._getExpFileString()
        outFile.write(toFile)
        outFile.close()

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
