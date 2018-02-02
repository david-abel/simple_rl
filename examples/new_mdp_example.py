#!/usr/bin/env python

# Python imports.
from collections import defaultdict
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.tasks import GridWorldMDP, GridWorldState
from simple_rl.run_experiments import run_agents_on_mdp 

class ColoredGridWorldMDP(GridWorldMDP):

    def __init__(self, col_sq_locs_dict, width=5, height=3, init_loc=(1, 1), goal_locs=[(5, 3)]):
        '''
        Args:
            col_sq_locs_dict (dict):
                Key: int (width)
                Val: dict
                    Key: int (height)
                    Val: color
            width (int)
            height (int)
            init_loc (tuple)
            goal_locs (list of tuples)
        '''
        GridWorldMDP.__init__(self,
                              width,
                              height,
                              init_loc=init_loc,
                              goal_locs=goal_locs,
                              init_state=ColoredGridWorldState(init_loc[0], init_loc[1], col_sq_locs_dict[init_loc[0]][init_loc[1]]))

        self.col_sq_locs_dict = col_sq_locs_dict

    def _transition_func(self, state, action):
        '''
        Args:
            state (ColoredGridWorldState)
            action (str)

        Returns:
            (ColoredGridWorldState)
        '''
        if state.is_terminal():
            return state

        if action == "up" and state.y < self.height:
            next_state = ColoredGridWorldState(state.x, state.y + 1, self.get_state_color(state.x, state.y + 1))
        elif action == "down" and state.y > 1:
            next_state = ColoredGridWorldState(state.x, state.y - 1, self.get_state_color(state.x, state.y - 1))
        elif action == "right" and state.x < self.width:
            next_state = ColoredGridWorldState(state.x + 1, state.y, self.get_state_color(state.x + 1, state.y))
        elif action == "left" and state.x > 1:
            next_state = ColoredGridWorldState(state.x - 1, state.y, self.get_state_color(state.x - 1, state.y))
        else:
            next_state = ColoredGridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) in self.goal_locs:
            next_state.set_terminal(True)

        return next_state

    def get_state_color(self, x, y):
        return self.col_sq_locs_dict[x][y]


class ColoredGridWorldState(GridWorldState):
    ''' Class for Colored Grid World States '''

    def __init__(self, x, y, color="white"):
        GridWorldState.__init__(self, x=x, y=y)
        self.color = color

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.color) + ")"

def main(open_plot=True):
    state_colors = defaultdict(lambda: defaultdict(lambda: "white"))
    state_colors[3][2] = "red"

    # Setup MDP, Agents.
    mdp = ColoredGridWorldMDP(state_colors)
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=15, episodes=500, steps=40, open_plot=open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
