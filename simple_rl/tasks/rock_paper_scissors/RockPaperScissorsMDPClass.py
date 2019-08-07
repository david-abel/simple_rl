''' RockPaperScissorsMDP.py: Contains an implementation of a two player Rock Paper Scissors game. '''

# Python imports.
import random

# Other imports
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from simple_rl.mdp.StateClass import State

class RockPaperScissorsMDP(MarkovGameMDP):
    ''' Class for a Rock Paper Scissors Game '''

    # Static constants.
    ACTIONS = ["rock", "paper", "scissors"]

    def __init__(self):
        MarkovGameMDP.__init__(self, RockPaperScissorsMDP.ACTIONS, self._transition_func, self._reward_func, init_state=State())

    def _reward_func(self, state, action_dict, next_state=None):
        '''
        Args:
            state (State)
            action (dict of actions)

        Returns
            (float)
        '''
        agent_a, agent_b = action_dict.keys()[0], action_dict.keys()[1]
        action_a, action_b = action_dict[agent_a], action_dict[agent_b]

        reward_dict = {}

        # Win conditions.
        a_win = (action_a == "rock" and action_b == "scissors") or \
                (action_a == "paper" and action_b == "rock") or \
                (action_a == "scissors" and action_b == "paper")

        if action_a == action_b:
            reward_dict[agent_a], reward_dict[agent_b] = 0, 0
        elif a_win:
            reward_dict[agent_a], reward_dict[agent_b] = 1, -1
        else:
            reward_dict[agent_a], reward_dict[agent_b] = -1, 1

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
        return "rock_paper_scissors"


def main():
    grid_world = RockPaperScissorsMDP()

if __name__ == "__main__":
    main()
