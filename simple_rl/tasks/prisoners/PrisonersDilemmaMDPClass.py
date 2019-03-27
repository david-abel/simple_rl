''' PrisonersDilemmaMDPClass.py: Contains an implementation of PrisonersDilemma. '''

# Python imports.
import random

# Other imports.
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from simple_rl.mdp.StateClass import State

class PrisonersDilemmaMDP(MarkovGameMDP):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["defect", "cooperate"]

    def __init__(self):
        MarkovGameMDP.__init__(self, PrisonersDilemmaMDP.ACTIONS, self._transition_func, self._reward_func, init_state=State())

    def _reward_func(self, state, action_dict, next_state=None):
        '''
        Args:
            state (State)
            action (dict of actions)
            next_state (State)

        Returns
            (float)
        '''
        agent_a, agent_b = action_dict.keys()[0], action_dict.keys()[1]
        action_a, action_b = action_dict[agent_a], action_dict[agent_b]

        reward_dict = {}

        if action_a == action_b == "cooperate":
            reward_dict[agent_a], reward_dict[agent_b] = 2, 2
        elif action_a == action_b == "defect":
            reward_dict[agent_a], reward_dict[agent_b] = 1, 1
        elif action_a == "cooperate" and action_b == "defect":
            reward_dict[agent_a] = 0
            reward_dict[agent_b] = 3
        elif action_a == "defect" and action_b == "cooperate":
            reward_dict[agent_a] = 3
            reward_dict[agent_b] = 0

        return reward_dict


    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action_dict (str)

        Returns
            (State)
        '''
        return state
      
    def __str__(self):
        return "prisoners_dilemma"


def main():
    grid_world = GridWorldMDP(5, 10, (1, 1), (6, 7))

if __name__ == "__main__":
    main()
