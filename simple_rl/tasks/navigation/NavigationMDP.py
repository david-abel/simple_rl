''' NavigationMDP.py: Contains the NavigationMDP class. '''

# Python imports.
from __future__ import print_function
import numpy as np
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class NavigationMDP(GridWorldMDP):
    
    '''
        Class for Navigation MDP from:
            MacGlashan, James, and Michael L. Littman. "Between Imitation and Intention Learning." IJCAI. 2015.
    '''

    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, 
                 width=30, 
                 height=30, 
                 init_loc=(1,1),
                 rand_init=True,
                 goal_locs=[(21, 21)],
                 cell_types=["empty", "yellow", "red", "green", "purple"],
                 cell_type_rewards=[0, 0, -10, -10, -10],
                 gamma=0.99,
                 slip_prob=0.00,
                 step_cost=0.0,
                 goal_reward=1.0,
                 is_goal_terminal=True,
                 init_state=None,
                 name="Navigation MDP"):
        """
        Note: 1. locations and state dimensions start from 1 instead of 0. 
              2. 2d locations are interpreted in (x,y) format.
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
            cell_type (list of cell types: [str, str, ...]): non-goal cell types
            cell_rewards (reward mapping for each cell type: [int, int, ...]): reward value for cells in @cell_type
        """
        
        assert height > 0 and isinstance(height, int) and width > 0 and isinstance(width, int), "height and widht must be integers and > 0"
        self.cell_types = cell_types

        # Probability of each cell type
        vacancy_prob = 0.8
        # Can't say more about these numbers (chose arbitrarily larger than percolation threshold for square lattice).
        # This is just an approximation as the paper isn't concerned about cell probabilities or mention it.
        self.cell_prob = [4.*vacancy_prob/5., vacancy_prob/5.] + [(1-vacancy_prob)/3.] * 3
        # Matrix for identifying cell type and associated reward
        self.cells = np.random.choice(len(self.cell_types), p=self.cell_prob, size=height*width).reshape(height,width)
        self.cell_type_rewards = cell_type_rewards
        self.cell_rewards = np.asarray([[cell_type_rewards[item] for item in row] for row in self.cells]).reshape(height,width)
        self.goal_reward = goal_reward

        GridWorldMDP.__init__(self, 
                            width=width, 
                            height=height,
                            init_loc=init_loc,
                            rand_init=rand_init,
                            goal_locs=goal_locs,
                            lava_locs=[()],
                            walls=[], # no walls in this mdp
                            is_goal_terminal=is_goal_terminal,
                            gamma=gamma,
                            init_state=init_state,
                            slip_prob=slip_prob,
                            step_cost=step_cost,
                            name=name)

        # Set goals and their rewards in the matrix
        for g in goal_locs:
            g_r, g_c = self._xy_to_rowcol(g[0], g[1])
            self.cells[g_r, g_c] = len(self.cell_types) # allocate the next type to the goal
            self.cell_rewards[g_r, g_c] = self.goal_reward
        self.goal_locs = goal_locs

        self.feature_cell_dist = None

    def get_cell_distance_features(self):

        if self.feature_cell_dist is not None:
            return self.feature_cell_dist

        self.loc_cells = [np.vstack(np.where(self.cells == cell)).transpose() for cell in range(len(self.cell_types))]
        self.feature_cell_dist = np.zeros(self.cells.shape + (len(self.cell_types),), np.float32)

        for row in range(self.height):
            for col in range(self.width):

                # Note: if particular cell type is missing in the grid, this will assign distance -1 to it
                self.feature_cell_dist[row, col] = [np.linalg.norm([row, col] - loc_cell, axis=1).min() \
                                                    if len(loc_cell) != 0 else -1 for loc_cell in self.loc_cells]
        return self.feature_cell_dist

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        r, c = self._xy_to_rowcol(state.x, state.y)
        if self._is_goal_state_action(state, action):
            return self.goal_reward + self.cell_rewards[r, c] - self.step_cost
        elif self.cell_rewards[r, c] == 0:
            return 0 - self.step_cost
        else:
            return self.cell_rewards[r, c]

    def _xy_to_rowcol(self, x, y):
        """
        Converts (x,y) to (row,col) 
        """
        return self.height - y, x - 1

    def _rowcol_to_xy(self, row, col):
        """
        Converts (row,col) to (x,y) 
        """
        return col + 1, self.height - row

    def get_random_init_state(self):
        """
        Returns a random empty/white cell 
        """
        rows, cols = np.where(self.cells == 0)
        rand_idx = np.random.randint(len(rows))
        x, y = self._rowcol_to_xy(rows[rand_idx], cols[rand_idx])
        return GridWorldState(x, y)

    def visualize_grid(self, values=None, cmap=None,
                                trajectories=None, subplot_str=None,
                                new_fig=True, show_rewards_cbar=False, title="Navigation MDP"):
        """
        Args:
            trajectories ([[state1, state2, ...], [state7, state4, ...], ...]): trajectories to be shown on the grid
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        cell_types = ['white', 'yellow', 'red', 'lime', 'magenta', 'blue']
        cell_type_rewards = self.cell_type_rewards + [self.goal_reward]

        if new_fig == True:
            plt.figure(figsize=(max(self.height // 4, 6), max(self.width // 4, 6)))

        if subplot_str is not None:
            plt.subplot(subplot_str)

        if cmap is None:
            cmap = colors.ListedColormap(cell_types)

        if values is None:
            values = self.cells.copy()

        im = plt.imshow(values, cmap=cmap)
        plt.title(title)
        ax = plt.gca()
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.set_xticks(np.arange(self.width), minor=True)
        ax.set_yticks(np.arange(self.height), minor=True)
        ax.set_xticklabels(1 + np.arange(self.width), minor=True, fontsize=8)
        ax.set_yticklabels(1 + np.arange(self.height)[::-1], minor=True, fontsize=8)

        if trajectories is not None and trajectories:
            for state_seq in trajectories:
                path_xs = [s.x - 1 for s in state_seq]
                path_ys = [self.height - (s.y) for s in state_seq]
                plt.plot(path_xs[0], path_ys[0], ".k", markersize=10)
                plt.plot(path_xs, path_ys, "k", linewidth=0.7)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)

        if show_rewards_cbar:
            cb = plt.colorbar(im, ticks=range(len(cell_type_rewards)), cax=cax)
            cb.set_ticklabels(cell_type_rewards)
        else:
            plt.colorbar(im, cax=cax)

        if subplot_str is None:
            plt.show()

