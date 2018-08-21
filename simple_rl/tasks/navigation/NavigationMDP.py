''' NavigationMDP.py: Contains the NavigationMDP class. '''

# Python imports.
from __future__ import print_function
import numpy as np
from collections import defaultdict
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

class NavigationMDP(GridWorldMDP):

    '''
        Class for Navigation MDP from:
            MacGlashan, James, and Michael L. Littman. "Between Imitation and 
            Intention Learning." IJCAI. 2015.
    '''

    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self,
                 width=30,
                 height=30,
                 goal_locs=[(21, 21)],
                 cell_types=["empty", "yellow", "red", "green", "purple"],
                 cell_type_rewards=[0, 0, -10, -10, -10],
                 cell_distribution="probability",
                 # cell_type_probs: default is chosen arbitrarily larger than
                 # percolation threshold for square lattice, which is just an
                 # approximation to match cell distribution with that of the
                 # paper.
                 cell_type_probs=[0.68, 0.17, 0.05, 0.05, 0.05],
                 cell_type_forced_locations=[np.inf, np.inf,
                                             [(1,1),(5,5)], [(2,2)], [4,4]],
                 gamma=0.99,
                 slip_prob=0.00,
                 step_cost=0.0,
                 goal_rewards=[1.0],
                 is_goal_terminal=True,
                 traj_init_cell_types=[0],
                 goal_colors=["blue"],
                 init_loc=(1,1),
                 rand_init=True,
                 init_state=None,
                 name="Navigation MDP"):
        """
        Note: 1. locations and state dimensions start from 1 instead of 0.
              2. 2d locations are interpreted in (x,y) format.
        Args:
            height (int): Height of navigation grid in no. of cells.
            width (int): Width of navigation grid in no. of cells.
            goal_locs (list of tuples: [(int, int)...]): Goal locations.
            cell_type (list of cell types: [str, str, ...]): Non-goal cell types.
            cell_rewards (list of ints): Reward for each @cell_type.
            cell_distribution (str): 
                "probability" - will assign cells according 
                to @cell_type_probs over the state space. 
                "manual" - will  use @cell_type_forced_locations to assign cells to locations.
            cell_type_probs (list of floats): Only applicable when
                @cell_distribution is set to "probability". Specifies probability 
                corresponding to each @cell_type. Values must sum to 1. Each value 
                signifies the probability of occurence of particular cell type in the grid.
                Note: the actual probabilities would be slightly off because 
                this doesn't factor in number of goals.
            cell_type_forced_locations (list of list of tuples 
            [[(x1,y1), (x2,y2)], [(x3,y3), ...], np.inf, ...}):
                Only applicable when @cell_distribution is set to "Manual". Used 
                to specify additional cells and their locations. If elements are 
                set to np.inf, all of them will be sampled uniformly at random.
            goal_colors (list of str/int): Color of goal corresponding to @goal_locs.
                If colors are different, each goal will be represented with 
                a unique feature, otherwise all goals are mapped to same feature.
            traj_init_cell_types (list of ints): To specify which cell types
                are navigable. This is used in sampling empty/drivable states 
                while generating trajectories.
        Not used but are properties of superclass GridWorldMDP:
            init_loc (tuple: (int, int)): (x,y) initial location
            rand_init (bool): Whether to use random initial location
            init_state (GridWorldState): Initial GridWorldState
            """

        assert height > 0 and isinstance(height, int) and width > 0 \
               and isinstance(width, int), "height and widht must be integers and > 0"
        assert len(goal_colors) == len(goal_locs) == len(goal_rewards)
        assert len(cell_types) == len(cell_type_rewards)
        assert cell_distribution == "manual" or len(cell_types) == len(cell_type_probs)
        assert cell_distribution == "probability" or len(cell_types) == len(cell_type_forced_locations)

        self.value_iter = None
        self._policy_invalidated = True
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

        # Cell Types
        self.cells = self.__generate_cell_type_grid(
                                            height, width,
                                            cell_distribution, cell_type_probs,
                                            cell_type_forced_locations)
        # Preserve a copy without goals
        self.cells_wo_goals = self.cells.copy()

        # Cell Rewards
        self.cell_type_rewards = cell_type_rewards
        self.cell_rewards = np.asarray(
                        [[self.cell_type_rewards[item] for item in row]
                            for row in self.cells]
                        ).reshape(height,width)
        # Preserve a copy without goals
        self.cell_rewards_wo_goals = self.cell_rewards.copy()

        # Update cells and cell_rewards with goal and its rewards
        self.reset_goals(goal_locs, goal_rewards, goal_colors)

        # Find set of Empty/Navigable cells for sampling trajectory init state
        self.set_traj_init_cell_types(cell_types=traj_init_cell_types)

        # Additional book-keeping
        self.feature_cell_dist = None
        self.feature_cell_dist_normalized = None


    def __generate_cell_type_grid(self, height, width, cell_distribution,
                            cell_type_probs, cell_type_forced_locations):

        assert cell_distribution in ["probability", "manual"]
        # Assign cell type over state space
        if cell_distribution == "probability":

            cells = np.random.choice(
                len(cell_type_probs),
                p=cell_type_probs,
                replace=True,
                size=(height, width))
        else:

            inf_cells = [idx for idx, elem in
                         enumerate(cell_type_forced_locations) if
                         elem == np.inf]
            if len(inf_cells) == 0:
                cells = -1 * np.ones((height, width), dtype=np.int)
            else:
                cells = np.random.choice(inf_cells,
                                              p=[1. / len(inf_cells)] * len(inf_cells),
                                              replace=True,
                                              size=(height, width))
            for cell_type, cell_locs in enumerate(cell_type_forced_locations):
                if cell_type not in inf_cells:
                    for cell_loc in cell_locs:
                        row, col = self._xy_to_rowcol(cell_loc[0], cell_loc[1])
                        cells[row, col] = cell_type

        # Additional check to ensure all states have corresponding cell type
        assert np.any(cells == -1) == False, \
            "Some grid cells have unassigned cell type! When you use manual " \
            "distribution, make sure each state of the MPD is covered by a " \
            "cell type. Check usage of np.inf in @cell_type_forced_locations."

        return cells

    def reset_goals(self, goal_locs, goal_rewards, goal_colors):
        """
        Resets the goals. Updates cell type grid and cell reward grid as per 
        new goal configuration.
        """

        # Maintain a copy in object
        self.goal_rewards = goal_rewards
        self.goal_locs = goal_locs
        self.goal_colors = goal_colors
        # Reset goal xy to idx dict
        self.goal_xy_to_idx = {}

        # Reset cell type and cell reward grid with no goals
        self.cells = self.cells_wo_goals.copy()
        self.cell_rewards = self.cell_rewards_wo_goals.copy()

        # Update goals and their rewards
        self.all_goals_state_features_entangled = len(np.unique(goal_colors)) == 1
        for goal_idx, goal_loc in enumerate(self.goal_locs):
            goal_r, goal_c = self._xy_to_rowcol(goal_loc[0], goal_loc[1])
            if self.all_goals_state_features_entangled:
                # Assume all goals are same
                self.cells[goal_r, goal_c] = len(self.cell_types)
                self.cell_rewards[goal_r, goal_c] = self.goal_rewards[0]
                self.goal_xy_to_idx[(goal_loc[0], goal_loc[1])] = 0
            else:
                # Each goal is different in type and rewards
                self.cells[goal_r, goal_c] = len(self.cell_types) + goal_idx
                self.cell_rewards[goal_r, goal_c] = self.goal_rewards[goal_idx]
                self.goal_xy_to_idx[(goal_loc[0], goal_loc[1])] = goal_idx
        self._policy_invalidated = True

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
            next_state = self._transition_func(state, action)
            return self.goal_rewards[self.goal_xy_to_idx[
                                    (next_state.x, next_state.y)]] \
                       + self.cell_rewards[r, c] - self.step_cost
        elif self.cell_rewards[r, c] == 0:
            return 0 - self.step_cost
        else:
            return self.cell_rewards[r, c] - self.step_cost

    def set_traj_init_cell_types(self, cell_types=[0]):

        self.traj_init_cell_row_idxs, self.traj_init_cell_col_idxs = [], []
        for cell_type in cell_types:
            rs, cs = np.where(self.cells == cell_type)
            self.traj_init_cell_row_idxs.extend(rs)
            self.traj_init_cell_col_idxs.extend(cs)
        self.num_traj_init_states = len(self.traj_init_cell_row_idxs)

    def sample_empty_state(self, idx=None):
        """
        Returns a random empty/white state of type GridWorldState()
        """

        if idx is None:
            rand_idx = np.random.randint(len(self.traj_init_cell_row_idxs))
        else:
            assert 0 <= idx < len(self.traj_init_cell_row_idxs)
            rand_idx = idx

        x, y = self._rowcol_to_xy(self.traj_init_cell_row_idxs[rand_idx],
                                  self.traj_init_cell_col_idxs[rand_idx])
        return GridWorldState(x, y)

    def sample_empty_states(self, n, repetition=False):
        """
        Returns a list of random empty/white state of type GridWorldState()
        Note: if repetition is False, the max no. of states returned = # of empty/white cells in the grid
        """
        assert n > 0

        if repetition is False:
            return [self.sample_empty_state(rand_idx) for rand_idx in np.random.permutation(len(self.traj_init_cell_row_idxs))[:n]]
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

    def compute_value_iteration_results(self, sample_rate):

        # If value iteration was run previously, don't re-run it
        if self.value_iter is None or self._policy_invalidated == True:
            self.value_iter = ValueIteration(self, sample_rate=sample_rate)
            _ = self.value_iter.run_vi()
            self._policy_invalidated = False
        return self.value_iter

    def sample_data(self, n_trajectory,
                    init_states=None,
                    init_repetition=False,
                    policy=None,
                    horizon=100,
                    pad_extra_trajectories=True,
                    value_iter_sampling_rate=1,
                    map_actions_to_index=True):
        """
        Args:
            n_trajectory: number of trajectories to sample
            init_states: 
                None - to use random init state 
                [GridWorldState(x,y),...] - to use specific init states
            init_repetition: When init_state is set to None, this will sample 
                every possible init state and try to not repeat init state 
                unless @n_trajectory > @self.num_traj_init_states
            policy (fn): S->A
            horizon (int): planning horizon
            pad_extra_trajectories: If True, this will always return 
                @n_trajectory many trajectories, overrides @init_repetition 
                if # unique states !=  @n_trajectory
            value_iter_sampling_rate (int): Used for value iteration if policy 
                is set to None
            map_actions_to_index (bool): Set True to get action indices in 
                trajectory
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
            if len(init_states) < n_trajectory and pad_extra_trajectories:
                init_states += self.sample_empty_states(n_trajectory - len(init_states), repetition=True)
        else:
            if len(init_states) < n_trajectory and pad_extra_trajectories:
                # More init states need to be sampled
                init_states += self.sample_empty_states(n_trajectory - len(init_states), init_repetition)
            else:
                # We have sufficient init states pre-specified, ignore the rest
                # as we only need n_trajectory many
                init_states = init_states[:n_trajectory]

        if policy is None:
            policy = self.compute_value_iteration_results(value_iter_sampling_rate).policy

        for init_state in init_states:
            action_seq, state_seq = self.plan(init_state, policy=policy, horizon=horizon)
            d_mdp_states.append(state_seq)
            if map_actions_to_index:
                a_s.append(action_seq)
            else:
                a_s.append([action_to_idx[a] for a in action_seq])
        return d_mdp_states, a_s

    def get_cell_distance_features(self,
                                   incl_goal_dist_feature=False,
                                   normalize=False):

        """
        Returns 3D array (x,y,z) where (x,y) refers to row and col of cells in the navigation grid and z is a vector of
        manhattan distance to each cell type.
        """
        if normalize and self.feature_cell_dist_normalized is not None:
            return self.feature_cell_dist_normalized
        elif normalize == False and self.feature_cell_dist is not None:
            return self.feature_cell_dist

        if incl_goal_dist_feature:
            # Include distance to goal
            if self.all_goals_state_features_entangled:
                dist_cell_types = range(0, len(self.cell_types)+1)
            else:
                dist_cell_types = range(0,
                                    len(self.cell_types)+len(self.goal_locs))
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

    def feature_at_loc(self,
                       x, y,
                       feature_type="indicator",
                       incl_distance_features=False,
                       incl_goal_ind_feature=False,
                       incl_goal_dist_feature=False,
                       normalize_distance=False,
                       dtype=np.float32):
        """
        Returns feature vector at a state corresponding to (x,y) location
        Args:
            x, y (int, int): cartesian coordinates in 2d state space
            feature_type (str): "indicator" to use one-hot encoding of cell type
                "cartesian" to use (x,y) and "rowcol" to use (row, col) as feature
            incl_distance_features (bool): True - appends feature vector with
                distance to each type of cell.
            incl_goal_ind_feature: True - adds goal indicator feature
                (only applicable for "indicator" feature type).
            normalize_distance (bool): Whether to normalize cell type distances
                 to 0-1 range (only used when "incl_distance_features" is True).
            dtype (numpy datatype): cast feature vector to dtype
        """
        row, col = self._xy_to_rowcol(x, y)
        assert feature_type in ["indicator", "cartesian", "rowcol"]

        if feature_type == "indicator":
            if incl_goal_ind_feature:
                if self.all_goals_state_features_entangled:
                    # i.e., all goals have same cell type (feature), so total no. of diff cel types = cell_types + 1
                    feature = np.eye(len(self.cell_types)+1)[self.cells[row, col]]
                else:
                    # Total no. of diff cel types = cell_types + no. of goals
                    feature = np.eye(len(self.cell_types) +
                                     len(self.goal_locs))[self.cells[row, col]]
            else:
                if (x, y) in self.goal_locs:
                    feature = np.zeros(len(self.cell_types))
                else:
                    feature = np.eye(len(self.cell_types))[self.cells[row, col]]
        elif feature_type == "cartesian":
            feature = np.array([x, y])
        elif feature_type == "rowcol":
            feature = np.array([row, col])

        if incl_distance_features:
            return np.hstack(
                    (feature,
                     self.get_cell_distance_features(
                             incl_goal_dist_feature, normalize_distance)[row, col])).astype(dtype)
        else:
            return feature.astype(dtype)

    def feature_at_state(self,
                         mdp_state,
                         feature_type="indicator",
                         incl_distance_features=False,
                         incl_goal_ind_feature=False,
                         incl_goal_dist_feature=False,
                         normalize_distance=False,
                         dtype=np.float32):
        """
        Returns feature vector at a state corresponding to MdP State
        Args:
            mdp_state (int, int): GridWorldState object
            feature_type (str): "indicator" to use one-hot encoding of cell type
                "cartesian" to use (x,y) and "rowcol" to use (row, col) as feature
            incl_distance_features (bool): True - appends feature vector with
                distance to each type of cell.
            incl_goal_ind_feature: True - adds goal indicator feature
                (only applicable for "indicator" feature type).
            normalize_distance (bool): Whether to normalize cell type distances
                to 0-1 range (only used when "incl_distance_features" is True).
            dtype (numpy datatype): cast feature vector to dtype
        """
        return self.feature_at_loc(mdp_state.x, mdp_state.y,
                                   feature_type,
                                   incl_distance_features,
                                   incl_goal_ind_feature,
                                   incl_goal_dist_feature,
                                   normalize_distance,
                                   dtype)

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
                            new_fig=True, show_rewards_cbar=False,
                            int_cells_cmap=cm.viridis,
                            title="Navigation MDP"):
        """
        Args:
            values (2d ndarray): Values to be visualized in the grid, defaults to cell types.
            cmap (Matplotlib Colormap): Colormap corresponding to values,
                defaults to ListedColormap with colors specified in "cell_types" during construction.
            trajectories ([[state1, state2, ...], [state7, state4, ...], ...]): trajectories to be shown on the grid.
            subplot_str (str): subplot number (e.g., "411", "412", etc.). Defaults to None.
            new_fig (Bool): Whether to use existing figure context. To show in subplot, set this to False.
            show_rewards_cbar (Bool): Whether to show colorbar with cell reward values.
            title (str): Title of the plot.
        """

        if self.all_goals_state_features_entangled:
            # If all goals are mapped to same feature, they'll all be shown with the same color
            cell_types = self.cell_types + [self.goal_colors[0]]
            cell_type_rewards = self.cell_type_rewards + [self.goal_rewards[0]]
        else:
            cell_types = self.cell_types + self.goal_colors
            cell_type_rewards = self.cell_type_rewards + self.goal_rewards

        if new_fig == True:
            plt.figure(figsize=(max(self.height // 4, 6), max(self.width // 4, 6)))

        if subplot_str is not None:
            plt.subplot(subplot_str)

        if cmap is None:
            norm = colors.Normalize(vmin=0, vmax=len(self.goal_colors)-1)
            # Leave string colors as it is, convert int colors to normalized rgba
            cell_colors = [int_cells_cmap(norm(cell))
                                if isinstance(cell, int)
                                    else cell
                                for cell in cell_types]
            cmap = colors.ListedColormap(cell_colors)

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
