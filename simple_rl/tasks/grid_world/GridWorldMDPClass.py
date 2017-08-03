''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
import random

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
                goal_locs=[(5,3)],
                walls=[],
                is_goal_terminal=True,
                gamma=0.99,
                init_state=None):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        init_state = GridWorldState(init_loc[0], init_loc[1]) if init_state is None else init_state
        MDP.__init__(self, GridWorldMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)
        if type(goal_locs) is not list:
            print "Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)]."
            quit()
        self.walls = walls
        for g in goal_locs:
            if g[0] > width or g[1] > height:
                print "Error: goal provided is off the map or overlaps with a wall.."
                print "\tGridWorld dimensions: (" + str(width) + "," + str(height) + ")"
                print "\tProblematic Goal:", g
                quit()
            if self.is_wall(g[0], g[1]):
                print "Error: goal provided is off the map or overlaps with a wall.."
                print "\tWalls:", walls
                print "\tProblematic Goal:", g
                quit()

        self.width = width
        self.height = height
        self.init_loc = init_loc
        self.goal_locs = goal_locs
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        self.is_goal_terminal = is_goal_terminal

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        _error_check(state, action)

        if self._is_goal_state_action(state, action):
            return 1.0
        else:
            return 0

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if (state.x, state.y) in self.goal_locs and self.is_goal_terminal:
            return False

        if action == "left" and (state.x - 1, state.y) in self.goal_locs:
            return True
        elif action == "right" and (state.x + 1, state.y) in self.goal_locs:
            return True
        elif action == "down" and (state.x, state.y - 1) in self.goal_locs:
            return True
        elif action == "up" and (state.x, state.y + 1) in self.goal_locs:
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
        if not self.is_goal_terminal:
            return "gridworld-no-term"
        return "gridworld_h-" + str(self.height) + "_w-" + str(self.width)

    def get_goal_locs(self):
        return self.goal_locs

    def visualize_policy(self, policy):
        from ...utils.mdp_visualizer import visualize_policy
        from grid_visualizer import _draw_state
        ["up", "down", "left", "right"]

        action_char_dict = {
            "up":u"\u2191",
            "down":u"\u2193",
            "left":u"\u2190",
            "right":u"\u2192"
        }

        visualize_policy(self, policy, _draw_state, action_char_dict)
        raw_input("Press anything to quit ")
        quit()


    def visualize_agent(self, agent):
        from ...utils.mdp_visualizer import visualize_agent
        from grid_visualizer import _draw_state
        visualize_agent(self, agent, _draw_state)
        raw_input("Press anything to quit ")
        quit()

    def visualize_value(self):
        from ...utils.mdp_visualizer import visualize_value
        from grid_visualizer import _draw_state
        visualize_value(self, _draw_state)
        raw_input("Press anything to quit ")
        quit()


def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in GridWorldMDP.ACTIONS:
        print "Error: the action provided (" + str(action) + ") was invalid."
        quit()

    if not isinstance(state, GridWorldState):
        print "Error: the given state (" + str(state) + ") was not of the correct class."
        quit()


def main():
    grid_world = GridWorldMDP(5, 10, (1, 1), (6, 7))

    grid_world.visualize()

if __name__ == "__main__":
    main()
