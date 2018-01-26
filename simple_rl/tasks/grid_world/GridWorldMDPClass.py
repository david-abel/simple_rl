''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
from __future__ import print_function
import random
import sys
import os
import numpy as np

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class GridWorldMDP(MDP):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self,
                width=5,
                height=3,
                init_loc=(1,1),
                rand_init=False,
                goal_locs=[(5,3)],
                lava_locs=[()],
                walls=[],
                is_goal_terminal=True,
                gamma=0.99,
                init_state=None,
                slip_prob=0.0,
                name="gridworld"):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
            lava_locs (list of tuples: [(int, int)...]): These locations return -1 reward.
        '''

        # Setup init location.
        self.rand_init = rand_init
        if rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            while init_loc in walls:
                init_loc = random.randint(1, width), random.randint(1, height)
        self.init_loc = init_loc
        init_state = GridWorldState(init_loc[0], init_loc[1]) if init_state is None or rand_init else init_state

        MDP.__init__(self, GridWorldMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

        if type(goal_locs) is not list:
            print("(simple_rl) GridWorld Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)].")
            quit()

        self.walls = walls
        self.width = width
        self.height = height
        self.goal_locs = goal_locs
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        self.is_goal_terminal = is_goal_terminal
        self.slip_prob = slip_prob
        self.name = name
        self.lava_locs = lava_locs

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if self._is_goal_state_action(state, action):
            return 1.0 - self.step_cost
        elif (state.x, state.y) in self.lava_locs:
            return -0.1
        else:
            return 0 - self.step_cost

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if (state.x, state.y) in self.goal_locs and self.is_goal_terminal:
            # Already at terminal.
            return False

        goals = self.goal_locs

        if action == "left" and (state.x - 1, state.y) in goals:
            return True
        elif action == "right" and (state.x + 1, state.y) in goals:
            return True
        elif action == "down" and (state.x, state.y - 1) in goals:
            return True
        elif action == "up" and (state.x, state.y + 1) in goals:
            return True
        else:
            return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state

        r = random.random()
        if self.slip_prob > r:
            # Flip dir.
            if action == "up":
                action = random.choice(["left", "right"])
            elif action == "down":
                action = random.choice(["left", "right"])
            elif action == "left":
                action = random.choice(["up", "down"])
            elif action == "right":
                action = random.choice(["up", "down"])

        if action == "up" and state.y < self.height and not self.is_wall(state.x, state.y + 1):
            next_state = GridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1 and not self.is_wall(state.x, state.y - 1):
            next_state = GridWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width and not self.is_wall(state.x + 1, state.y):
            next_state = GridWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1 and not self.is_wall(state.x - 1, state.y):
            next_state = GridWorldState(state.x - 1, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) in self.goal_locs and self.is_goal_terminal:
            next_state.set_terminal(True)

        return next_state

    def is_wall(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''
        
        return (x, y) in self.walls

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def get_goal_locs(self):
        return self.goal_locs

    def visualize_policy(self, policy):
        from simple_rl.utils import mdp_visualizer as mdpv
        from grid_visualizer import _draw_state
        ["up", "down", "left", "right"]

        action_char_dict = {
            "up":u"\u2191",
            "down":u"\u2193",
            "left":u"\u2190",
            "right":u"\u2192"
        }

        mdpv.visualize_policy(self, policy, _draw_state, action_char_dict)
        raw_input("Press anything to quit ")
        quit()


    def visualize_agent(self, agent):
        from simple_rl.utils import mdp_visualizer as mdpv
        from grid_visualizer import _draw_state
        mdpv.visualize_agent(self, agent, _draw_state)
        raw_input("Press anything to quit ")

    def visualize_value(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from grid_visualizer import _draw_state
        mdpv.visualize_value(self, _draw_state)
        raw_input("Press anything to quit ")


def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in GridWorldMDP.ACTIONS:
        print("(simple_rl) GridWorldError: the action provided (" + str(action) + ") was invalid in state: " + str(state) + ".")
        quit()

    if not isinstance(state, GridWorldState):
        print("(simple_rl) GridWorldError: the given state (" + str(state) + ") was not of the correct class.")
        quit()


def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0):
    '''
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)

    Returns:
        (GridWorldMDP)

    Summary:
        Builds a GridWorldMDP from a file:
            'w' --> wall
            'a' --> agent
            'g' --> goal
            '-' --> empty
    '''

    if name is None:
        name = file_name.split(".")[0]

    grid_path = os.path.dirname(os.path.realpath(__file__))
    wall_file = open(os.path.join(grid_path, "txt_grids", file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    empty_cells = []
    agent_x, agent_y = 1, 1
    walls = []
    goal_locs = []
    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_x, agent_y = j + 1, num_rows - i
            elif ch == "-":
                empty_cells.append((j + 1, num_rows - i))

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]]

    if randomize:
        agent_x, agent_y = random.choice(empty_cells)
        if len(goal_locs) == 0:
            # Sample @num_goals random goal locations.
            goal_locs = random.sample(empty_cells, num_goals)
        else:
            goal_locs = random.sample(goal_locs, num_goals)

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

    return GridWorldMDP(width=num_cols, height=num_rows, init_loc=(agent_x, agent_y), goal_locs=goal_locs, walls=walls, name=name, slip_prob=slip_prob)

    def reset(self):
        if self.rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        else:
            self.cur_state = copy.deepcopy(self.init_state)


def main():
    grid_world = GridWorldMDP(5, 10, (1, 1), (6, 7))

    grid_world.visualize()

if __name__ == "__main__":
    main()
