''' NavigationMDP.py: Contains the NavigationMDP class. '''

# Python imports.
from __future__ import print_function
import numpy as np
from collections import defaultdict
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration
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
                 additional_obstacles={}, # this is for additional experimentation only
                 gamma=0.99,
                 slip_prob=0.00,
                 step_cost=0.0,
                 goal_reward=1.0,
                 is_goal_terminal=True,
                 init_state=None,
                 vacancy_prob=0.8,
                 sample_cell_types=[0],
                 use_goal_dist_feature=True,
                 goal_color="blue",
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
        GridWorldMDP.__init__(self,

                              width=width,
                              height=height,
                              init_loc=init_loc,
                              rand_init=rand_init,
                              goal_locs=goal_locs,
                              lava_locs=[()],
                              walls=[],  # no walls in this mdp
                              is_goal_terminal=is_goal_terminal,
                              gamma=gamma,
                              init_state=init_state,
                              slip_prob=slip_prob,
                              step_cost=step_cost,
                              name=name)

        # Probability of each cell type
        if len(additional_obstacles) > 0:
            self.cell_prob = np.zeros(len(self.cell_types))
            self.cell_prob[0] = 1.
        else:
            # Can't say more about these numbers (chose arbitrarily larger than percolation threshold for
            # square lattice). This is just an approximation (to match cell distribution with that of the paper);
            # however, it is not the primary concern here.
            self.cell_prob = [8.*vacancy_prob/10., 2*vacancy_prob/10.] + [(1-vacancy_prob)/3.] * 3

        # Matrix for identifying cell type and associated reward
        self.cells = np.random.choice(len(self.cell_types), p=self.cell_prob, size=height*width).reshape(height,width)

        self.additional_obstacles = additional_obstacles
        for obs_type, obs_locs in self.additional_obstacles.items():
            for obs_loc in obs_locs:
                row, col = self._xy_to_rowcol(obs_loc[0], obs_loc[1])
                self.cells[row, col] = obs_type

        self.cell_type_rewards = cell_type_rewards
        self.cell_rewards = np.asarray([[cell_type_rewards[item] for item in row] for row in self.cells]).reshape(height,width)
        self.goal_reward = goal_reward

        # Set goals and their rewards in the matrix
        for g in goal_locs:
            g_r, g_c = self._xy_to_rowcol(g[0], g[1])
            self.cells[g_r, g_c] = len(self.cell_types) # allocate the next type to the goal
            self.cell_rewards[g_r, g_c] = self.goal_reward
        self.goal_locs = goal_locs
        self.use_goal_dist_feature = use_goal_dist_feature
        self.goal_color = goal_color
        self.feature_cell_dist = None
        self.feature_cell_dist_normalized = None
        self.value_iter = None
        self.define_sample_cells(cell_types=sample_cell_types)

    def define_sample_cells(self, cell_types=[0]):

        self.sample_rows, self.sample_cols = [], []

        for cell_type in cell_types:
            rs, cs= np.where(self.cells == cell_type)
            self.sample_rows.extend(rs)
            self.sample_cols.extend(cs)
        self.num_empty_states = len(self.sample_rows)

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

    def get_value_iteration_results(self, sample_rate):
        if self.value_iter is None:
            self.value_iter = ValueIteration(self, sample_rate=sample_rate)
            _ = self.value_iter.run_vi()
        return self.value_iter

    def sample_empty_state(self, idx=None):
        """
        Returns a random empty/white state of type GridWorldState()
        """

        if idx is None:
            rand_idx = np.random.randint(len(self.sample_rows))
        else:
            assert 0 <= idx < len(self.sample_rows)
            rand_idx = idx

        x, y = self._rowcol_to_xy(self.sample_rows[rand_idx], self.sample_cols[rand_idx])
        return GridWorldState(x, y)

    def sample_empty_states(self, n, repetition=False):
        """
        Returns a list of random empty/white state of type GridWorldState()
        Note: if repetition is False, the max no. of states returned = # of empty/white cells in the grid 
        """
        assert n > 0

        if repetition is False:
            return [self.sample_empty_state(rand_idx) for rand_idx in np.random.permutation(len(self.sample_rows))[:n]]
        else:
            return [self.sample_empty_state() for i in range(n)]

    def plan(self, state, policy=None, horizon=100):
        '''
        Args:
            state (State)
            policy (fn): S->A
            horizon (int)

        Returns:
            (list): List of actions
        '''
        action_seq = []
        state_seq = [state]
        steps = 0

        while (not state.is_terminal()) and steps < horizon:
            next_action = policy(state)
            action_seq.append(next_action)
            state = self.transition_func(state, next_action)
            state_seq.append(state)
            steps += 1

        return action_seq, state_seq

    def sample_data(self, n_trajectory,
                    init_states=None,
                    init_repetition=False,
                    policy=None,
                    horizon=100,
                    pad_to_match_n_trajectory=True,
                    value_iter_sampling_rate=1):
        """
        Args:
            n_trajectory: number of trajectories to sample
            init_state: None - to use random init state [GridWorldState(x,y),...] - to use specific init states 
            init_repetition: When init_state is set to None, this will sample every possible init state 
                                    and try to not repeat init state unless n_trajectory > n_states
            policy (fn): S->A
            horizon (int): planning horizon
            pad_to_match_n_trajectory: If True, this will always return n_trajectory many trajectories 
                                        overrides init_repetition if # unique states !=  n_trajectory
            value_iter_sampling_rate (int): Used for value iteration if policy is set to None
                                    
        Returns:
            (Traj_states, Traj_actions) where
                Traj_states: [[s1, s2, ..., sT], [s4, s1, ..., sT], ...],
                Traj_actions: [[a1, a2, ..., aT], [a4, a1, ..., aT], ...]
        """
        a_s = []
        d_mdp_states = []
        visited_at_init = defaultdict(lambda: False)
        action_to_idx = {a: i for i, a in enumerate(self.actions)}

        if init_states is None:
            init_states = self.sample_empty_states(n_trajectory, init_repetition)
            if len(init_states) < n_trajectory and pad_to_match_n_trajectory:
                init_states += self.sample_empty_states(n_trajectory - len(init_states), repetition=True)
        else:
            if len(init_states) < n_trajectory and pad_to_match_n_trajectory: # i.e., more init states need to be sampled
                init_states += self.sample_empty_states(n_trajectory - len(init_states), init_repetition)
            else: # we have sufficient init states pre-specified, ignore the rest as we only need n_trajectory many
                init_states = init_states[:n_trajectory]

        if policy is None:
            policy = self.get_value_iteration_results(value_iter_sampling_rate).policy

        for init_state in init_states:
            action_seq, state_seq = self.plan(init_state, policy=policy, horizon=horizon)
            d_mdp_states.append(state_seq)
            a_s.append([action_to_idx[a] for a in action_seq])
        return d_mdp_states, a_s

    def get_cell_distance_features(self, normalize=True):

        """
        Returns 3D array (x,y,z) where (x,y) refers to row and col of cells in the navigation grid and z is a vector of 
        manhattan distance to each cell type.     
        """
        if normalize and self.feature_cell_dist_normalized is not None:
            return self.feature_cell_dist_normalized
        elif normalize == False and self.feature_cell_dist is not None:
            return self.feature_cell_dist

        if self.use_goal_dist_feature:
            # +1 to include goal and start at 1 to ignore white/empty cell
            dist_cell_types = range(0, len(self.cell_types)+1)
        else:
            dist_cell_types = range(0, len(self.cell_types))

        self.loc_cells = [np.vstack(np.where(self.cells == cell)).transpose() for cell in dist_cell_types]
        self.feature_cell_dist = np.zeros(self.cells.shape + (len(dist_cell_types),), np.float32)

        for row in range(self.height):
            for col in range(self.width):

                # Note: if particular cell type is missing in the grid, this will assign distance -1 to it
                # Ord=1: Manhattan, Ord=2: Euclidean and so on
                self.feature_cell_dist[row, col] = [np.linalg.norm([row, col] - loc_cell, ord=1, axis=1).min() \
                                                    if len(loc_cell) != 0 else -1 for loc_cell in self.loc_cells]

        # feature scaling
        if normalize:
            max_dist = self.width + self.height
            self.feature_cell_dist_normalized = self.feature_cell_dist / max_dist

        return self.feature_cell_dist

    def feature_short_at_state(self, mdp_state, normalize=True):
        return self.feature_short_at_loc(mdp_state.x, mdp_state.y, normalize)

    def feature_long_at_state(self, mdp_state, normalize=True):
        return self.feature_long_at_loc(mdp_state.x, mdp_state.y, normalize)

    def feature_short_at_loc(self, x, y, normalize=True):
        row, col = self._xy_to_rowcol(x, y)
        if (x, y) in self.goal_locs:
            return np.zeros(len(self.cell_types), dtype=np.float32)
        else:
            return np.eye(len(self.cell_types))[self.cells[row, col]]

    def feature_long_at_loc(self, x, y, normalize=True):
        row, col = self._xy_to_rowcol(x, y)
        return np.hstack((self.feature_short_at_loc(x, y, normalize), self.get_cell_distance_features(normalize)[row, col]))

    def states_to_features(self, states, phi):
        return np.asarray([phi(s) for s in states], dtype=np.float32)

    def states_to_coord(self, states, phi):
        return np.asarray([(s.x, s.y) for s in states], dtype=np.float32)

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

        cell_types = self.cell_types + [self.goal_color]
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
                plt.plot(path_xs, path_ys, "k", linewidth=0.7)
                plt.plot(path_xs[0], path_ys[0], ".k", markersize=10)
                plt.plot(path_xs[-1], path_ys[-1], "*c", markersize=10)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)

        if show_rewards_cbar:
            cb = plt.colorbar(im, ticks=range(len(cell_type_rewards)), cax=cax)
            cb.set_ticklabels(cell_type_rewards)
        else:
            plt.colorbar(im, cax=cax)

        if subplot_str is None:
            plt.show()
