# Python imports.
import sys
import os
import time
sys.path.append(os.getcwd() + "/../agents/")
sys.path.append(os.getcwd() + "/../tasks/grid_world/")
from collections import defaultdict

# Local imports.
from GridWorldMDPClass import GridWorldMDP
from ExperimentClass import Experiment
from RandomAgentClass import RandomAgent
from RMaxAgentClass import RMaxAgent
from QLearnerAgentClass import QLearnerAgent

def runAgentsOnMDP(agents, mdp, numInstances=5, numEpisodes=100, numSteps=100):

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
                state = mdp.getCurState()
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
                mdp.reset()

            # Process that learning instance's info at end of learning.
            experiment.endOfInstance(agent)

            # Reset the agent and MDP.
            agent.reset()
        end = time.clock()

        # Track how much time this agent took.
        times[agent] = round(end - start,3)

    # Time stuff.
    print "--- TIMES ---"
    for agent in times.keys():
        print str(agent) + " agent took " + str(times[agent]) + " seconds."
    print "-------------"

    experiment.makePlots()


def main():
    # MDP.
    gw = GridWorldMDP(5,5, (1,1), (5,5))

    # Agent.
    randomAgent = RandomAgent(actions = GridWorldMDP.actions)
    rMaxAgent = RMaxAgent(actions = GridWorldMDP.actions)
    qLearnerAgent = QLearnerAgent(actions = GridWorldMDP.actions)

    # Run experiments.
    runAgentsOnMDP([qLearnerAgent, randomAgent], gw)


if __name__ == "__main__":
    main()