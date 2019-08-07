''' GridGameMDPClass.py: Contains an implementation of a two player grid game. '''

# Python imports.
import random

# Other imports.
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from simple_rl.tasks.grid_game.GridGameStateClass import GridGameState

class GridGameMDP(MarkovGameMDP):
    ''' Class for a Two Player Grid Game '''

    # Static constants.
    ACTIONS = ["up", "left", "down", "right"]

    def __init__(self, height=3, width=8, init_a_x=1, init_a_y=2, init_b_x=8, init_b_y=8):
        self.goal_a_x = init_b_x
        self.goal_a_y = init_b_y
        self.goal_b_x = init_a_x
        self.goal_b_y = init_a_y
        init_state = GridGameState(init_a_x, init_a_y, init_b_x, init_b_y)
        self.height = height
        self.width = width
        MarkovGameMDP.__init__(self, GridGameMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state)

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["width"] = self.width
        param_dict["height"] = self.height
        param_dict["init_a_x"] = self.init_a_x
        param_dict["init_a_y"] = self.init_a_y
        param_dict["init_b_x"] = self.init_b_x
        param_dict["init_b_y"] = self.init_b_y
   
        return param_dict
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

        next_state = self._transition_func(state, action_dict)

        a_at_goal = (next_state.a_x == self.goal_a_x and next_state.a_y == self.goal_a_y)
        b_at_goal = (next_state.b_x == self.goal_b_x and next_state.b_y == self.goal_b_y)
        
        if a_at_goal and b_at_goal:
            reward_dict[agent_a] = 2.0
            reward_dict[agent_b] = 2.0
        elif a_at_goal and not b_at_goal:
            reward_dict[agent_a] = 1.0
            reward_dict[agent_b] = -1.0
        elif not a_at_goal and b_at_goal:
            reward_dict[agent_a] = -1.0
            reward_dict[agent_b] = 1.0
        else:
            reward_dict[agent_a] = 0.0
            reward_dict[agent_b] = 0.0

        return reward_dict

    def _transition_func(self, state, action_dict):
        '''
        Args:
            state (State)
            action_dict (str)

        Returns
            (State)
        '''
        
        agent_a, agent_b = action_dict.keys()[0], action_dict.keys()[1]
        action_a, action_b = action_dict[agent_a], action_dict[agent_b]

        next_state = self._move_agents(action_a, state.a_x, state.a_y, action_b, state.b_x, state.b_y)

        return next_state
      
    def _move_agents(self, action_a, a_x, a_y, action_b, b_x, b_y):
        '''
        Args:
            action_a (str)
            a_x (int)
            a_y (int)
            action_b (str)
            b_x (int)
            b_y (int)

        Summary:
            Moves the two agents accounting for collisions with walls and each other.

        Returns:
            (GridGameState)
        '''

        new_a_x, new_a_y = a_x, a_y
        new_b_x, new_b_y = b_x, b_y

        # Move agent a.
        if action_a == "up" and a_y < self.height:
            new_a_y += 1
        elif action_a == "down" and a_y > 1:
            new_a_y -= 1
        elif action_a == "right" and a_x < self.width:
            new_a_x += 1
        elif action_a == "left" and a_x > 1:
            new_a_x -= 1

        # Move agent b.
        if action_b == "up" and b_y < self.height:
            new_b_y += 1
        elif action_b == "down" and b_y > 1:
            new_b_y -= 1
        elif action_b == "right" and b_x < self.width:
            new_b_x += 1
        elif action_b == "left" and b_x > 1:
            new_b_x -= 1
        
        if new_a_x == new_b_x and new_a_y == new_b_y or \
            (new_a_x == b_x and new_a_y == b_y and new_b_x == a_x and new_b_y == a_y):
            # If the agent's collided or traded places, reset them.
            new_a_x, new_a_y = a_x, a_y
            new_b_x, new_b_y = b_x, b_y

        next_state = GridGameState(new_a_x, new_a_y, new_b_x, new_b_y)

        # Check terminal.
        if self._is_terminal_state(next_state):
            next_state.set_terminal(True)

        return next_state

    def _is_terminal_state(self, next_state):
        return (next_state.a_x == self.goal_a_x and next_state.a_y == self.goal_a_y) or \
            (next_state.b_x == self.goal_b_x and next_state.b_y == self.goal_b_y)

    def __str__(self):
        return "grid_game-" + str(self.height) + "-" + str(self.width)

def _manhattan_distance(a_x, a_y, b_x, b_y):
    return abs(a_x - b_x) + abs(a_y - b_y)

def main():
    grid_game = GridGameMDP()

if __name__ == "__main__":
    main()
