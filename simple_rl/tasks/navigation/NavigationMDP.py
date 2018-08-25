''' NavigationMDP.py: Contains the NavigationMDP class. '''
# Python imports.
from __future__ import print_function
import copy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class NavigationMDP(GridWorldMDP):

    '''
        Class for Navigation MDP from:
            MacGlashan, James, and Michael L. Littman. "Between Imitation and
            Intention Learning." IJCAI. 2015.
    '''

    ACTIONS = ["up", "down", "left", "right"]

    @staticmethod
    def states_to_features(states, phi):
        """
        Returns phi(states)    
        """
        return np.asarray([phi(s) for s in states], dtype=np.float32)

    @staticmethod
    def states_to_coord(states, phi=None):
        """
        Returns phi(states)    
        """
        return np.asarray([(s.x, s.y) for s in states], dtype=np.float32)

    def __init__(self, width=30, height=30,
                 living_cell_types=["empty", "yellow", "red", "green", "purple"],
                 living_cell_rewards=[0, 0, -10, -10, -10],
                 living_cell_distribution="probability",
                 living_cell_type_probs=[0.68, 0.17, 0.05, 0.05, 0.05],
                 living_cell_locs=[np.inf, np.inf, [(1,1),(5,5)], [(2,2)], [4,4]],
                 goal_cell_locs=[],
                 goal_cell_rewards=[],
                 goal_cell_types=[],
                 gamma=0.99, slip_prob=0.00, step_cost=0.0,
                 is_goal_terminal=True, traj_init_cell_types=[0],
                 planning_init_loc=(1,1), planning_rand_init=True, name="Navigation MDP"):
        """
        Note: Locations are specified in (x,y) format, but (row, col) convention 
            is used while storing in memory. 
        Args:
            height (int): Height of navigation grid in no. of cells.
            width (int): Width of navigation grid in no. of cells.
            living_cell_types (list of cell types: [str, str, ...]): Non-goal cell types.
            living_cell_rewards (list of int): Reward for each @cell_type.
            living_cell_distribution (str):
                "probability" - will assign cells according to @living_cell_type_probs.
                "manual" - uses @living_cell_locs to assign cells to state space.
            living_cell_type_probs (list of floats): Probability corresponding to 
                each @living_cell_types. 
                Note: Goal isn't factored so actual probabilities can off.
                Default values are chosen arbitrarily larger than percolation threshold 
                for square lattice, which is just an approximation to match cell 
                distribution with that of the paper.
            living_cell_locs (list of list of tuples
            [[(x1,y1), (x2,y2)], [(x3,y3), ...], np.inf, ...}):
                Specifies living cell locations. If elements are set to np.inf, 
                they will be sampled uniformly at random.
            goal_cell_locs (list of tuples: [(int, int)...]): Goal locations.
            goal_cell_rewards (list of int): Goal rewards.
            goal_cell_types (list of str/int): Type of goal corresponding to @goal_cell_locs.
            traj_init_cell_types (list of int): Trajectory init state sampling cell type 
            """
        assert height > 0 and isinstance(height, int) and width > 0 \
               and isinstance(width,
                              int), "height and widht must be integers and > 0"
        assert len(living_cell_types) == len(living_cell_rewards)
        assert living_cell_distribution == "manual" or len(living_cell_types) == len(
            living_cell_type_probs)
        assert living_cell_distribution == "probability" or len(
            living_cell_types) == len(living_cell_locs)
        assert len(goal_cell_types) == len(goal_cell_locs) == len(
            goal_cell_rewards)

        GridWorldMDP.__init__(self, width=width, height=height,
                              init_loc=planning_init_loc,
                              rand_init=planning_rand_init,
                              goal_locs=goal_cell_locs, lava_locs=[()],
                              walls=[], is_goal_terminal=is_goal_terminal,
                              gamma=gamma, init_state=None,
                              slip_prob=slip_prob, step_cost=step_cost,
                              name=name)

        # Sets up state space (2d grid where each element holds a cell id)
        self.__setup_state_space(height, width, living_cell_types,
                                    living_cell_distribution,
                                    living_cell_type_probs, living_cell_locs,
                                    goal_cell_types, goal_cell_locs)
        # Sets up rewards over state space
        self._reset_rewards(living_cell_rewards, goal_cell_rewards)

        # Find set of Empty/Navigable cells for sampling trajectory init state
        self.set_traj_init_cell_types(cell_types=traj_init_cell_types)

        # Initialize value iteration object (computes reachable states)
        self.value_iter = ValueIteration(self, sample_rate=1)

        # Additional book-keeping
        self.feature_cell_dist = None
        self.feature_cell_dist_kind = 0

    def __setup_state_space(self, height, width, living_cell_types,
                            living_cell_distribution,
                            living_cell_type_probs, living_cell_locs,
                            goal_cell_types, goal_cell_locs):

        # Enumerate living and goal cell type ids
        self.living_cell_types = living_cell_types
        self.goal_cell_types = goal_cell_types
        self.living_cell_ids = list(range(len(living_cell_types)))
        self.goal_cell_ids = list(range(self.living_cell_ids[-1] + 1,
                                        self.living_cell_ids[-1] + 1 + len(
                                            goal_cell_locs)))
        # Combined types and ids
        self.cell_types = self.living_cell_types + self.goal_cell_types
        self.cell_ids = self.living_cell_ids + self.goal_cell_ids

        # Initialize state space with living cells
        self.state_space = self.__get_living_cells_to_state_space(height, width,
                                                                  living_cell_types,
                                                                  living_cell_distribution,
                                                                  living_cell_type_probs,
                                                                  living_cell_locs)
        # Preserve a copy without goals
        self.state_space_wo_goals = self.state_space.copy()

        # Add goals to state space
        self.goal_cell_locs = goal_cell_locs
        self.state_space, self.goal_xy_to_idx = self.__add_goal_cells_to_state_space(
            self.state_space, goal_cell_locs, self.goal_cell_ids)

    def __add_goal_cells_to_state_space(self, state_space, goal_cell_locs,
                                        goal_cell_ids):

        # Goal xy to idx dict
        goal_xy_to_idx = {}

        # Add goal cells
        for idx, goal_loc in enumerate(goal_cell_locs):
            goal_r, goal_c = self._xy_to_rowcol(goal_loc[0], goal_loc[1])
            state_space[goal_r, goal_c] = goal_cell_ids[idx]
            goal_xy_to_idx[(goal_loc[0], goal_loc[1])] = idx
        return state_space, goal_xy_to_idx

    def __get_living_cells_to_state_space(self, height, width,
                                          living_cell_types,
                                          living_cell_distribution,
                                          living_cell_type_probs,
                                          living_cell_locs):

        assert living_cell_distribution in ["probability", "manual"]
        # Assign cell type over state space
        if living_cell_distribution == "probability":

            cells = np.random.choice(len(living_cell_types),
                                     p=living_cell_type_probs,
                                     replace=True, size=(height, width))
        else:

            inf_cells = [idx for idx, elem in enumerate(living_cell_locs) if
                         elem == np.inf]
            if len(inf_cells) == 0:
                cells = -1 * np.ones((height, width), dtype=np.int)
            else:
                cells = np.random.choice(inf_cells,
                                         p=[1. / len(inf_cells)] * len(
                                             inf_cells),
                                         replace=True, size=(height, width))

            for cell_type, cell_locs in enumerate(living_cell_locs):
                if cell_type not in inf_cells:
                    for cell_loc in cell_locs:
                        row, col = self._xy_to_rowcol(cell_loc[0], cell_loc[1])
                        cells[row, col] = cell_type

        # Additional check to ensure all states have corresponding cell type
        assert np.any(cells == -1) == False, \
            "Some grid cells have unassigned cell type! When you use manual " \
            "distribution, make sure each state of the MPD is covered by a " \
            "cell type. Check usage of np.inf in @living_cell_locs."
        return cells

    def _reset_rewards(self, living_cell_rewards, goal_cell_rewards):
        """
        Sets up rewards grid corresponding to @self.state_space
        """
        self.living_cell_rewards = living_cell_rewards
        self.goal_cell_rewards = goal_cell_rewards
        self.cell_type_rewards = self.living_cell_rewards + self.goal_cell_rewards

        # State rewards with no goals (preserve a copy without goals)
        self.state_rewards_wo_goals = np.asarray(
            [[living_cell_rewards[item] for item in row] for row in
             self.state_space_wo_goals]).reshape(self.height, self.width)
        # State rewards
        self.state_rewards = np.asarray(
            [[self.cell_type_rewards[item] for item in row] for row in
             self.state_space]).reshape(self.height, self.width)

        # Mark flag for invalidating previous value iteration results
        self._policy_invalidated = True

    def _reset_goals(self, goal_cell_locs, goal_cell_rewards, goal_cell_types):
        """
        Resets the goals. Updates cell type grid and cell reward grid as per
        new goal configuration.
        """
        # Reset goal cell type ids
        self.goal_cell_types = goal_cell_types
        self.goal_cell_ids = list(range(self.living_cell_ids[-1] + 1,
                                        self.living_cell_ids[-1] + 1 + len(
                                            goal_cell_locs)))
        self.cell_types = self.living_cell_types + self.goal_cell_types
        self.cell_ids = self.living_cell_ids + self.goal_cell_ids

        # Retrieve state space copy without goals
        self.state_space = self.state_space_wo_goals.copy()
        # Add goals to state space
        self.goal_cell_locs = goal_cell_locs
        self.state_space, self.goal_xy_to_idx = self.__add_goal_cells_to_state_space(
            self.state_space, goal_cell_locs, self.goal_cell_ids)

        # Redefine rewards over state space
        self._reset_rewards(self.living_cell_rewards, goal_cell_rewards)

        # Mark flag for invalidating previous value iteration results
        self._policy_invalidated = True

    def get_states(self):
        """
        Returns all reachable states 
        """
        return self.value_iter.get_states()

    def get_trans_dict(self):
        """
        Returns transition dynamics matrix 
        """
        self.value_iter._compute_matrix_from_trans_func()
        return self.value_iter.trans_dict

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
            return self.goal_cell_rewards[self.goal_xy_to_idx[
                (next_state.x, next_state.y)]] \
                   + self.state_rewards[r, c] - self.step_cost
        elif self.state_rewards[r, c] == 0:
            return 0 - self.step_cost
        else:
            return self.state_rewards[r, c] - self.step_cost

    def set_traj_init_cell_types(self, cell_types=[0]):
        """
        Sets cell types for sampling first state of trajectory 
        """
        self.traj_init_cell_row_idxs, self.traj_init_cell_col_idxs = [], []
        for cell_type in cell_types:
            rs, cs = np.where(self.state_space == cell_type)
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

    def sample_init_states(self, n, repetition=False):
        """
        Returns a list of random empty/white state of type GridWorldState()
        Note: if repetition is False, the max no. of states returned = # of empty cells in the grid
        """
        assert n > 0

        if repetition is False:
            return [self.sample_empty_state(rand_idx) for rand_idx in
                    np.random.permutation(len(self.traj_init_cell_row_idxs))[
                    :n]]
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

    def run_value_iteration(self):
        """
        Runs value iteration (if needed) and returns ValueIteration object.
        """
        # If value iteration was run previously, don't re-run it
        if self._policy_invalidated == True:
            _ = self.value_iter.run_vi()
            self._policy_invalidated = False
        return self.value_iter

    def get_value_grid(self):
        """
        Returns value over states space grid
        """
        value_iter = self.run_value_iteration()
        V = np.zeros((self.height, self.width), dtype=np.float32)
        for row in range(self.height):
            for col in range(self.width):
                x, y = self._rowcol_to_xy(row, col)
                V[row, col] = value_iter.value_func[GridWorldState(x, y)]
        return V

    def sample_data(self, n_trajectory, init_states=None,
                    init_repetition=False, policy=None, horizon=100,
                    pad_extra_trajectories=True, map_actions_to_index=True):
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
            init_states = self.sample_init_states(n_trajectory, init_repetition)
            if len(init_states) < n_trajectory and pad_extra_trajectories:
                init_states += self.sample_init_states(n_trajectory - len(init_states), repetition=True)
        else:
            if len(init_states) < n_trajectory and pad_extra_trajectories:
                # More init states need to be sampled
                init_states += self.sample_init_states(n_trajectory - len(init_states), init_repetition)
            else:
                # We have sufficient init states pre-specified, ignore the rest
                # as we only need n_trajectory many
                init_states = init_states[:n_trajectory]

        if policy is None:
            if len(self.goal_cell_locs) == 0:
                print("Running value iteration with no goals assigned..")
            policy = self.run_value_iteration().policy

        for init_state in init_states:
            action_seq, state_seq = self.plan(init_state, policy=policy, horizon=horizon)
            d_mdp_states.append(state_seq)
            if map_actions_to_index:
                a_s.append(action_seq)
            else:
                a_s.append([action_to_idx[a] for a in action_seq])
        return d_mdp_states, a_s

    def __transfrom(self, mat, type):
        if type == "normalize_manhattan":
            return mat / (self.width + self.height)
        if type == "normalize_euclidean":
            return mat / np.sqrt(self.width**2 + self.height**2)
        else:
            return mat

    def compute_grid_distance_features(self, incl_cells, incl_goals, normalize=False):
        """
        Computes distances to specified cell types for entire grid. 
        Returns 3D array (row,col,distance)
        """
        # Convert 2 flags to decimal representation. This is useful to check if
        # requested features are different from those previously stored
        feature_kind = int(str(int(incl_cells)) + str(int(incl_goals)), 2)
        if self.feature_cell_dist is not None \
                and self.feature_cell_dist_kind == feature_kind:
            return self.__transfrom(self.feature_cell_dist,
                                    "normalize_manhattan" if normalize else "None")
        dist_cell_types = copy.deepcopy(self.living_cell_ids) if incl_cells else []
        dist_cell_types += self.goal_cell_ids if incl_goals else []

        loc_cells = [
            np.vstack(np.where(self.state_space == cell)).transpose() for cell
            in dist_cell_types]
        self.feature_cell_dist = np.zeros(
            self.state_space.shape + (len(dist_cell_types),), np.float32)
        for row in range(self.height):
            for col in range(self.width):
                # Note: if particular cell type is missing in the grid, this
                # will assign distance -1 to it
                # Ord=1: Manhattan, Ord=2: Euclidean and so on
                self.feature_cell_dist[row, col] = [
                    np.linalg.norm([row, col] - loc_cell, ord=1, axis=1).min()
                    if len(loc_cell) != 0 else -1 for loc_cell in loc_cells]

        self.feature_cell_dist_kind = feature_kind
        return self.__transfrom(self.feature_cell_dist,
                                "normalize_manhattan" if normalize else "None")

    def feature_at_loc(self, x, y, feature_type="indicator",
                       incl_cell_distances=False, incl_goal_indicator=False,
                       incl_goal_distances=False, normalize_distance=False,
                       dtype=np.float32):
        """
        Returns feature vector at a state corresponding to (x,y) location
        Args:
            x, y (int, int): cartesian coordinates in 2d state space
            feature_type (str): "indicator" to use one-hot encoding of cell type
                "cartesian" to use (x,y) and "rowcol" to use (row, col) as feature
            incl_cell_distances (bool): Include distance to each type of cell.
            incl_goal_distances (bool): Include distance to the goals.
            incl_goal_indicator: True - adds goal indicator feature
                If all goals have same color, it'll add single indicator variable 
                for all goals, otherwise it'll use different indicator variables 
                for each goal.
                (only applicable for "indicator" feature type).
            normalize_distance (bool): Whether to normalize cell type distances
                 to 0-1 range (only used when "incl_distance_features" is True).
            dtype (numpy datatype): cast feature vector to dtype
        """
        row, col = self._xy_to_rowcol(x, y)
        assert feature_type in ["indicator", "cartesian", "rowcol"]

        if feature_type == "indicator":
            if incl_goal_indicator:
                ind_feature = np.eye(len(self.cell_ids))[self.state_space[row, col]]
            else:
                if (x, y) in self.goal_cell_locs:
                    ind_feature = np.zeros(len(self.living_cell_ids))
                else:
                    ind_feature = np.eye(len(self.living_cell_ids))[self.state_space[row, col]]
        elif feature_type == "cartesian":
            ind_feature = np.array([x, y])
        elif feature_type == "rowcol":
            ind_feature = np.array([row, col])

        if incl_cell_distances or incl_goal_distances:
            return np.hstack((ind_feature,
                              self.compute_grid_distance_features(
                                incl_cell_distances, incl_goal_distances,
                                normalize_distance)[row, col])).astype(dtype)
        else:
            return ind_feature.astype(dtype)

    def feature_at_state(self, mdp_state, feature_type="indicator",
                         incl_cell_distances=False, incl_goal_indicator=False,
                         incl_goal_distances=False, normalize_distance=False,
                         dtype=np.float32):
        """
        Returns feature vector at a state corresponding to MdP State
        Args:
            mdp_state (int, int): GridWorldState object
            feature_type (str): "indicator" to use one-hot encoding of cell type
                "cartesian" to use (x,y) and "rowcol" to use (row, col) as feature
            incl_cell_distances (bool): Include distance to each type of cell.
            incl_goal_distances (bool): Include distance to the goals.
            incl_goal_indicator: True - adds goal indicator feature
                If all goals have same color, it'll add single indicator variable 
                for all goals, otherwise it'll use different indicator variables 
                for each goal.
                (only applicable for "indicator" feature type).
            normalize_distance (bool): Whether to normalize cell type distances
                to 0-1 range (only used when "incl_distance_features" is True).
            dtype (numpy datatype): cast feature vector to dtype
        """
        return self.feature_at_loc(mdp_state.x, mdp_state.y, feature_type,
                                   incl_cell_distances, incl_goal_indicator,
                                   incl_goal_distances, normalize_distance,
                                   dtype)

    def __display_text(self, ax, x_start, x_end, y_start, y_end, values,
                       fontsize=12):
        """
        Ref: https://stackoverflow.com/questions/33828780/matplotlib-display-array-values-with-imshow
        """
        x_size = x_end - x_start
        y_size = y_end - y_start
        x_positions = np.linspace(start=x_start, stop=x_end, num=x_size,
                                  endpoint=False)
        y_positions = np.linspace(start=y_start, stop=y_end, num=y_size,
                                  endpoint=False)
        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = values[y_index, x_index]
                ax.text(x, y, label, color='black', ha='center',
                        va='center', fontsize=fontsize)

    def visualize_grid(self, values=None, cmap=cm.viridis, trajectories=None,
                       subplot_str=None, new_fig=True, show_colorbar=False,
                       show_rewards_colorbar=False, state_space_cmap=True,
                       init_marker=".k", traj_marker="-k",
                       text_values=None, text_size=10,
                       traj_linewidth=0.7, init_marker_sz=10,
                       goal_marker="*c", goal_marker_sz=10,
                       end_marker="", end_marker_sz=10,
                       axis_tick_font_sz=8, title="Navigation MDP"):
        """
        Args:
            values (2d ndarray): Values to be visualized in the grid, 
                defaults to cell types.
            cmap (Matplotlib Colormap): Colormap corresponding to values,
                defaults to ListedColormap with colors specified in 
                @self.living_cell_types and @self.goal_cell_types
            trajectories: Trajectories to be shown on the grid.
            subplot_str (str): Subplot number string (e.g., "411", "412", etc.)
            new_fig (bool): Whether to use existing figure context.
            show_rewards_colorbar (bool): Shows colorbar with cell reward values.
            title (str): Title of the plot.
        """
        if new_fig == True:
            plt.figure(
                figsize=(max(self.height // 4, 6), max(self.width // 4, 6)))
        # Subplot if needed
        if subplot_str is not None:
            plt.subplot(subplot_str)

        # Use state space (cell types) if values is None
        if values is None:
            values = self.state_space.copy()

        # Colormap
        if cmap is not None and state_space_cmap:
            norm = colors.Normalize(vmin=0, vmax=len(self.cell_types)-1)
            # Leave string colors as it is, convert int colors to normalized rgba
            cell_colors = [
                cmap(norm(cell)) if isinstance(cell, int) else cell
                for cell in self.cell_types]
            cmap = colors.ListedColormap(cell_colors)

        # Plot values
        im = plt.imshow(values, interpolation='None', cmap=cmap)
        plt.title(title)
        ax = plt.gca()
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.set_xticks(np.arange(self.width), minor=True)
        ax.set_yticks(np.arange(self.height), minor=True)
        ax.set_xticklabels(1 + np.arange(self.width), minor=True,
                           fontsize=axis_tick_font_sz)
        ax.set_yticklabels(1 + np.arange(self.height)[::-1], minor=True,
                           fontsize=axis_tick_font_sz)
        # Plot Trajectories
        if trajectories is not None and len(trajectories) > 0:
            for state_seq in trajectories:
                if len(state_seq) == 0:
                    continue
                path_xs = [s.x - 1 for s in state_seq]
                path_ys = [self.height - (s.y) for s in state_seq]
                plt.plot(path_xs, path_ys, traj_marker, linewidth=traj_linewidth)
                plt.plot(path_xs[0], path_ys[0], init_marker,
                         markersize=init_marker_sz) # Mark init state
                plt.plot(path_xs[-1], path_ys[-1], end_marker,
                         markersize=end_marker_sz) # Mark end state
        # Mark goals
        if len(self.goal_cell_locs) != 0:
            for goal_x, goal_y in self.goal_cell_locs:
                plt.plot(goal_x - 1, self.height - goal_y, goal_marker,
                         markersize=goal_marker_sz)
        # Text values on cell
        if text_values is not None:
            self.__display_text(ax, 0, self.width, 0, self.height, text_values,
                                fontsize=text_size)
        # Colorbar
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            if show_rewards_colorbar:
                cb = plt.colorbar(im, ticks=range(len(self.cell_type_rewards)), cax=cax)
                cb.set_ticklabels(self.cell_type_rewards)
            else:
                plt.colorbar(im, cax=cax)
        if subplot_str is None:
            plt.show()
