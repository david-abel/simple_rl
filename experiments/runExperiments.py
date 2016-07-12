# Python imports.
import sys
import os
import time
sys.path.append(os.getcwd() + "/../agents/")
sys.path.append(os.getcwd() + "/../tasks/grid_world/")
sys.path.append(os.getcwd() + "/../tasks/chain/")
from collections import defaultdict

# Local imports.
from GridWorldMDPClass import GridWorldMDP
from ChainMDPClass import ChainMDP
from ExperimentClass import Experiment
from RandomAgentClass import RandomAgent
from RMaxAgentClass import RMaxAgent
from QLearnerAgentClass import QLearnerAgent

def runAgentsOnMDP(agents, mdp, numInstances=3, numEpisodes=1000, numSteps=50):

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
    # gw = GridWorldMDP(5,5, (1,1), (5,5))
    chain = ChainMDP(15)

    # Agent.
    randomAgent = RandomAgent(ChainMDP.actions)
    rMaxAgent = RMaxAgent(actions = ChainMDP.actions)
    qLearnerAgent = QLearnerAgent(actions = ChainMDP.actions)

    # Run experiments.
    runAgentsOnMDP([qLearnerAgent, randomAgent], chain)


if __name__ == "__main__":
    main()