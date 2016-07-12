'''
Code for running experiments on an MDP

Instructions:
    (1) Set mdp in main.
    (2) Create agent instances.
    (3) Set experiment parameters (numInstances, numEpisodes, numSteps).
    (4) Call runAgentsOnMDP.

Author: David Abel
'''

# Python imports.
import sys
import os
import time
from collections import defaultdict

# Local imports.
from tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from tasks.chain.ChainMDPClass import ChainMDP
from experiments.ExperimentClass import Experiment
from agents.RandomAgentClass import RandomAgent
from agents.RMaxAgentClass import RMaxAgent
from agents.QLearnerAgentClass import QLearnerAgent

def runAgentsOnMDP(agents, mdp, numInstances=200, numEpisodes=25, numSteps=40):

    # Experiment (for reproducibility, plotting).
    experiment = Experiment(agents=agents, mdp=mdp, params = {"numInstances":numInstances, "numEpisodes":numEpisodes, "numSteps":numSteps})

    times = defaultdict(float)

    # Learn.
    for agent in agents:
        print str(agent) + " is learning."
        start = time.clock()

        # For each instance of the agent.
        for instance in xrange(numInstances):
            # For each episode.
            for episode in xrange(numEpisodes):

                # Compute initial state/reward.
                state = mdp.getInitState()
                reward = 0

                for step in xrange(numSteps):
                    # Compute the agent's policy.
                    action = agent.act(state, reward)

                    # Execute the action in the MDP.
                    reward, nextState = mdp.executeAgentAction(action)

                    # Record the experience.
                    experiment.addExperience(agent, state, action, reward, nextState)

                    # Update pointer.
                    state = nextState

                # Process experiment info at end of episode.
                experiment.endOfEpisode(agent)

            # Process that learning instance's info at end of learning.
            experiment.endOfInstance(agent)

            # Reset the agent and MDP.
            agent.reset()
        end = time.clock()

        # Track how much time this agent took.
        times[agent] = round(end - start,3)

    # Time stuff.
    print "\n--- TIMES ---"
    for agent in times.keys():
        print str(agent) + " agent took " + str(times[agent]) + " seconds."
    print "-------------\n"

    experiment.makePlots()


def main():
    # MDP.
    mdp = GridWorldMDP(5,5, (1,1), (5,5))
    # mdp = ChainMDP(15)
    actions = mdp.getActions()

    # Agent.
    randomAgent = RandomAgent(actions)
    rMaxAgent = RMaxAgent(actions)
    qLearnerAgent = QLearnerAgent(actions)

    # Run experiments.
    runAgentsOnMDP([qLearnerAgent, randomAgent], mdp)


if __name__ == "__main__":
    main()