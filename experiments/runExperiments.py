# Python imports.
import sys
import os
sys.path.append(os.getcwd() + "/../agents/")
sys.path.append(os.getcwd() + "/../grid_world/")

# Local imports.
from GridWorldClass import GridWorld
from RandomAgentClass import RandomAgent
from ExperimentClass import Experiment


def runAgentsOnMDP(agents, mdp, numSteps):

    # Experiment (for reproducibility, plotting).
    experiment = Experiment(agents=agents, mdp=mdp, params = {"numSteps":numSteps})

    # Learn.
    state = mdp.getCurState()
    for agent in agents:
        for t in range(numSteps):
            action = agent.policy(state)
            nextState, reward = mdp.executeAgentAction(action)
            experiment.addExperience(agent, state, action, reward, nextState)

    experiment.makePlots()


def main():
    # MDP.
    gw = GridWorld(10,10, (1,1), (10,10))

    # Agent.
    randomAgent = RandomAgent(GridWorld.actions)

    # Run experiments.
    runAgentsOnMDP([randomAgent], gw, numSteps=1000)


if __name__ == "__main__":
    main()