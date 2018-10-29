from __future__ import print_function
import copy
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.navigation.NavigationStateClass import NavigationWorldState
from simple_rl.planning import ValueIteration


class NavigationWorldMDP(MDP):
    """Class for Navigation MDP from:

            MacGlashan, James, and Michael L. Littman. "Between Imitation and
            Intention Learning." IJCAI. 2015.
    """
    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]
    CELL_KIND_NAV = "nav"
    CELL_KIND_WALL = "wall"
    CELL_KIND_GOAL = "goal"

    def __init__(self,
                 width=30,
                 height=30,
                 nav_cell_types=["white", "yellow", "red", "green", "purple"],
                 nav_cell_rewards=[0, 0, -10, -10, -10],
                 nav_cell_p_or_locs=[0.68, 0.17, 0.05, 0.05, 0.05],
                 wall_cell_types=[],
                 wall_cell_rewards=[],
                 wall_cell_locs=[],
                 goal_cell_types=["blue"],
                 goal_cell_rewards=[1.],
                 goal_cell_locs=[[(21, 21)]],
                 init_loc=(1, 1),
                 rand_init=False,
                 gamma=0.99, slip_prob=0.00, step_cost=0.5,
                 is_goal_terminal=True,
                 name="Navigation MDP"):
        """Navigation World MDP constructor.

        Args:
            height (int): Navigation grid height (in no. of cells).
            width (int): Navigation grid width (in no. of cells).
            nav_cell_types (list of <str / int>): Navigation cell types.
            nav_cell_rewards (list of float): Rewards associated with
                @nav_cell_types.
            nav_cell_p_or_locs (list of <float /
            list of tuples (int x, int y)>):
                Probability corresponding to @nav_cell_types distribution, or
                it could be list of fixed/forced locations [(x,y),...].
                (Default values are chosen arbitrarily larger than percolation
                threshold for square lattice--just an approximation to match
                cell distribution in the paper.)
            goal_cell_types (list of <str / int>): Goal cell types.
            goal_cell_locs (list of list of tuples (int, int)): Goal cell
                locations [(x,y),...] associated with @goal_cell_types.
            nav_cell_rewards (list of float): Rewards associated with
                @goal_cell_types.
            init_loc (int x, int y): Init cell to compute state space
                reachability and value iteration.
            gamma (float): MDP discount factor
            slip_prob (float): With this probability agent could fall either
                on left or right from the intended cell.
            step_cost (float): Living penalty.
            is_goal_terminal (bool): True to set goal state terminal.
        Note:
            Locations are specified in (x,y) format, but (row, col)
            convention is used while storing in memory.
        """
        assert len(nav_cell_types) == len(nav_cell_rewards) \
            == len(nav_cell_p_or_locs)
        assert len(wall_cell_types) == len(wall_cell_rewards) \
            == len(wall_cell_locs)
        assert len(goal_cell_types) == len(goal_cell_rewards) \
            == len(goal_cell_locs)

        self.width = width
        self.height = height
        self.init_loc = init_loc
        self.rand_init = rand_init
        self.gamma = gamma
        self.slip_prob = slip_prob
        self.step_cost = step_cost
        self.is_goal_terminal = is_goal_terminal
        self.name = name
        self.goal_cell_locs = goal_cell_locs

        # Setup init location.
        self.rand_init = rand_init
        if rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            while self.is_wall(*init_loc) or self.is_goal(*init_loc):
                init_loc = random.randint(1, width), random.randint(1, height)
        self.init_loc = init_loc
        # Construct base class
        MDP.__init__(self, NavigationWorldMDP.ACTIONS, self._transition_func,
                     self._reward_func,
                     init_state=NavigationWorldState(*init_loc),
                     gamma=gamma)

        # Navigation MDP
        self.__reset_nav_mdp()
        self.__register_cell_types(nav_cell_types, wall_cell_types,
                                   goal_cell_types)
        self.__add_cells_by_locs(self.goal_cell_ids, goal_cell_locs,
                                 NavigationWorldMDP.CELL_KIND_GOAL)
        self.__add_cells_by_locs(self.wall_cell_ids, wall_cell_locs,
                                 NavigationWorldMDP.CELL_KIND_WALL)
        self.__add_nav_cells(self.nav_cell_ids, nav_cell_p_or_locs,
                             NavigationWorldMDP.CELL_KIND_NAV)
        self.__check_state_map()

        self.__register_cell_rewards(nav_cell_rewards,
                                     wall_cell_rewards, goal_cell_rewards)

        # Initialize value iteration object (computes reachable states)
        self.value_iter = ValueIteration(self, sample_rate=1)
        self._policy_invalidated = True

    # ---------------------
    # -- Navigation MDP --
    # ---------------------
    def _xy_to_rowcol(self, x, y):
        """Converts (x, y) to (row, col).

        """
        return self.height - y, x - 1

    def _rowcol_to_xy(self, row, col):
        """Converts (row, col) to (x, y).

        """
        return col + 1, self.height - row

    def __reset_cell_type_params(self):
        self.__max_cell_id = -1
        self.combined_cell_types = []
        self.combined_cell_ids = []
        self.cell_type_to_id = {}
        self.cell_id_to_type = {}

    def __reset_nav_mdp(self):

        self.__reset_cell_type_params()
        self.num_traj_init_states = None
        self.feature_cell_dist = None
        self.feature_cell_dist_kind = 0
        self.nav_p_cell_ids = []
        self.nav_p_cell_probs = []
        self.__reset_state_map()

    def __reset_state_map(self):
        self.map_state_cell_id = -1 * np.ones((self.height, self.width),
                                              dtype=np.int)
        self.xy_to_cell_kind = defaultdict(lambda: "<Undefined>")

    def __check_state_map(self):

        if np.any(self.map_state_cell_id == -1):
            raise ValueError("Some states have unassigned cell type!"
                             "Make sure each state of the MDP is covered by"
                             "a cell type. Check usage of probability values"
                             "in @nav_cell_p_or_locs.")

    def __assign_cell_ids(self, cell_types):

        n = len(cell_types)
        cell_ids = list(
            range(self.__max_cell_id + 1, self.__max_cell_id + 1 + n))
        for cell_type, cell_id in zip(cell_types, cell_ids):
            self.cell_type_to_id[cell_type] = cell_id
            self.cell_id_to_type[cell_id] = cell_type
        self.combined_cell_types += cell_types
        self.combined_cell_ids += cell_ids
        self.__max_cell_id += n
        return cell_ids

    def __register_cell_types(self, nav_cell_types, wall_cell_types,
                              goal_cell_types):

        self.__reset_cell_type_params()
        self.nav_cell_types = nav_cell_types
        self.wall_cell_types = wall_cell_types
        self.goal_cell_types = goal_cell_types

        self.nav_cell_ids = self.__assign_cell_ids(nav_cell_types)
        self.wall_cell_ids = self.__assign_cell_ids(wall_cell_types)
        self.goal_cell_ids = self.__assign_cell_ids(goal_cell_types)
        self.n_unique_cells = self.__max_cell_id + 1

    def __add_cells_by_locs(self, cell_ids, cell_locs_list,
                            kind="<Undefined>"):

        for idx, cell_id in enumerate(cell_ids):
            cell_locs = cell_locs_list[idx]
            assert isinstance(cell_locs, list)
            for x, y in cell_locs:
                r, c = self._xy_to_rowcol(x, y)
                self.map_state_cell_id[r, c] = cell_id
                self.xy_to_cell_kind[(x, y)] = kind

    def __add_nav_cells(self, cell_ids, cell_p_or_locs_list,
                        kind="<Undefined>"):

        self.nav_p_cell_ids = []
        self.nav_p_cell_probs = []

        for idx, cell_id in enumerate(cell_ids):
            if isinstance(cell_p_or_locs_list[idx], list):  # locations
                cell_locs = cell_p_or_locs_list[idx]
                for x, y in cell_locs:
                    r, c = self._xy_to_rowcol(x, y)
                    if self.map_state_cell_id[r, c] == -1:
                        self.map_state_cell_id[r, c] = cell_id
                        self.xy_to_cell_kind[(x, y)] = kind
            else:
                assert isinstance(cell_p_or_locs_list[idx],
                                  float)  # probability values
                prob = cell_p_or_locs_list[idx]
                self.nav_p_cell_ids.append(cell_id)
                self.nav_p_cell_probs.append(prob)

        assert round(sum(self.nav_p_cell_probs),
                     9) == 1, "Probability values must sum to 1."
        for r in range(self.height):
            for c in range(self.width):
                if self.map_state_cell_id[r, c] == -1:
                    self.map_state_cell_id[r, c] = np.random.choice(
                        self.nav_p_cell_ids,
                        size=1,
                        p=self.nav_p_cell_probs)
                    x, y = self._rowcol_to_xy(r, c)
                    self.xy_to_cell_kind[(x, y)] = kind

    def __register_cell_rewards(self, nav_cell_rewards,
                                wall_cell_rewards, goal_cell_rewards):
        self.nav_cell_rewards = nav_cell_rewards
        self.wall_cell_rewards = wall_cell_rewards
        self.goal_cell_rewards = goal_cell_rewards
        self.cell_type_rewards = nav_cell_rewards + \
            wall_cell_rewards + \
            goal_cell_rewards

    def get_cell_id(self, x, y):
        """Get cell id of (x, y) location.

        Returns:
            A unique id assigned to each cell type.
        """
        return self.map_state_cell_id[tuple(self._xy_to_rowcol(x, y))]

    def is_wall(self, x, y):
        """Checks if (x,y) cell is wall or not.

        Returns:
            (bool): True iff (x, y) is a wall location.
        """
        return self.get_state_kind(x, y) == NavigationWorldMDP.CELL_KIND_WALL

    def is_goal(self, x, y):
        """Checks if (x,y) cell is goal or not.

        Returns:
            (bool): True iff (x, y) is a goal location.
        """
        return self.get_state_kind(x, y) == NavigationWorldMDP.CELL_KIND_GOAL

    def get_state_kind(self, x, y):
        """Returns kind of state at (x, y) location.

        Returns:
            (str): "wall", "nav", or "goal".
        """
        return self.xy_to_cell_kind[(x, y)]

    def _reset_goals(self, goal_cell_locs, goal_cell_rewards, goal_cell_types):
        """Resets the goals.

        Resamples previous goal locations with navigation cells.
        """
        # Re-sample old goal state cells
        old_goal_locs = self.goal_cell_locs
        # To keep it simple, navigation cells are sampled uniformly at random
        # to replace existing goal cells.
        new_cell_ids = np.random.choice(self.nav_cell_ids, size=len(old_goal_locs))
        self.__add_cells_by_locs(new_cell_ids, old_goal_locs,
                                 NavigationWorldMDP.CELL_KIND_NAV)

        self.__register_cell_types(self.nav_cell_types, self.wall_cell_types,
                                   goal_cell_types)
        self.__add_cells_by_locs(self.goal_cell_ids, goal_cell_locs,
                                 NavigationWorldMDP.CELL_KIND_GOAL)
        self.__check_state_map()
        self.__register_cell_rewards(self.nav_cell_rewards,
                                     self.wall_cell_rewards, goal_cell_rewards)
        self._policy_invalidated = True

    def _reset_rewards(self, nav_cell_rewards, wall_cell_rewards,
                       goal_cell_rewards):
        """Resets rewards corresponding to navigation, wall, and goal cells.

        """
        self.__register_cell_rewards(nav_cell_rewards, wall_cell_rewards,
                                     goal_cell_rewards)
        self._policy_invalidated = True

    # ---------
    # -- MDP --
    # ---------
    def _is_goal_state_action(self, state, action):
        """
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to
            the goal state.

        """
        if self.is_goal(state.x, state.y) == "goal" and self.is_goal_terminal:
            # Already at terminal.
            return False

        if action == "left" and self.is_goal(state.x - 1, state.y):
            return True
        elif action == "right" and self.is_goal(state.x + 1, state.y):
            return True
        elif action == "down" and self.is_goal(state.x, state.y - 1):
            return True
        elif action == "up" and self.is_goal(state.x, state.y + 1):
            return True
        else:
            return False

    def _transition_func(self, state, action):
        """
        Args:
            state (State)
            action (str)

        Returns
            (State)
        """
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

        if action == "up" and state.y < self.height and not self.is_wall(
                state.x, state.y + 1):
            next_state = NavigationWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1 and not self.is_wall(
                state.x, state.y - 1):
            next_state = NavigationWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width and not self.is_wall(
                state.x + 1, state.y):
            next_state = NavigationWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1 and not self.is_wall(state.x - 1,
                                                                   state.y):
            next_state = NavigationWorldState(state.x - 1, state.y)
        else:
            next_state = NavigationWorldState(state.x, state.y)

        if self.is_goal(next_state.x, next_state.y) and self.is_goal_terminal:
            next_state.set_terminal(True)

        return next_state

    def _reward_func(self, state, action):
        """
        Args:
            state (State)
            action (str)

        Returns
            (float)
        """
        next_state = self._transition_func(state, action)
        return self.cell_type_rewards[
            self.get_cell_id(next_state.x, next_state.y)] - self.step_cost

    # -------------------------
    # -- Trajectory Sampling --
    # -------------------------
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

    def set_traj_init_cell_types(self, cell_types):
        """
        Sets cell types for sampling first state of trajectory
        """
        self.traj_init_cell_row_idxs, self.traj_init_cell_col_idxs = [], []
        for cell_type in cell_types:
            rs, cs = np.where(self.map_state_cell_id ==
                              self.cell_type_to_id[cell_type])
            self.traj_init_cell_row_idxs.extend(rs)
            self.traj_init_cell_col_idxs.extend(cs)
        self.num_traj_init_states = len(self.traj_init_cell_row_idxs)

    def sample_traj_init_state(self, idx=None):
        """Samples trajectory init state (GridWorldState).

        Type of init cells to be sampled can be specified using
        set_traj_init_cell_types().
        """
        if idx is None:
            rand_idx = np.random.randint(len(self.traj_init_cell_row_idxs))
        else:
            assert 0 <= idx < len(self.traj_init_cell_row_idxs)
            rand_idx = idx

        x, y = self._rowcol_to_xy(self.traj_init_cell_row_idxs[rand_idx],
                                  self.traj_init_cell_col_idxs[rand_idx])
        return NavigationWorldState(x, y)

    def sample_init_states(self, n, init_unique=False, skip_states=None):
        """Returns a list of init states (GridWorldState).

        If init_unique is True, the max no. of states returned =
        # of empty cells in the grid.
        """
        assert n > 0

        if init_unique:
            c = 0
            init_states_list = []
            for idx in np.random.permutation(
                    len(self.traj_init_cell_row_idxs)):
                state = self.sample_traj_init_state(idx)
                if skip_states is None or state not in skip_states:
                    init_states_list.append(state)
                    c += 1
                    if c == n:
                        return init_states_list
            return init_states_list
        else:
            return [self.sample_traj_init_state() for i in range(n)]

    def sample_trajectories(self, n_traj, horizon, init_states=None,
                            init_cell_types=None, init_unique=False,
                            policy=None, rand_init_to_match_n_traj=True):
        """Samples trajectories.

        Args:
            n_traj: Number of trajectories to sample.
            horizon (int): Planning horizon (max trajectory length).
            init_states:
                None - to use random init state
                [GridWorldState(x,y),...] - to use specific init states
            init_unique: When init_unique is set to False, this will sample
                every possible init state and try to not repeat init state
                unless @n_traj > @self.num_traj_init_states
            policy (fn): S->A
            rand_init_to_match_n_traj: If True, this will always return
                @n_traj many trajectories. If # of unique states are less
                than @n_traj, this will override the effect of @init_unique and
                sample repeated init states.
        Returns:
            (traj_states_list, traj_actions_list) where
                traj_states: [s1, s2, ..., sT]
                traj_actions: [a1, a2, ..., aT]
        """
        assert len(init_cell_types) >= 1

        self.set_traj_init_cell_types(init_cell_types)
        traj_states_list = []
        traj_action_list = []
        traj_init_states = []

        if init_states is None:
            # If no init_states are provided, sample n_traj many init states.
            traj_init_states = self.sample_init_states(n_traj,
                                                       init_unique=init_unique)
        else:
            traj_init_states = copy.deepcopy(init_states)

        if len(traj_init_states) >= n_traj:
            traj_init_states = traj_init_states[:n_traj]
        else:
            # If # init_states < n_traj, sample remaining ones
            if len(traj_init_states) < n_traj:
                traj_init_states += self.sample_init_states(
                    n_traj - len(traj_init_states), init_unique=init_unique,
                    skip_states=traj_init_states)

            # If rand_init_to_match_n_traj is set to True, sample more
            # init_states if needed (may not be unique)
            if rand_init_to_match_n_traj and len(traj_init_states) < n_traj:
                traj_init_states += self.sample_init_states(n_traj,
                                                            init_unique=False)

        if policy is None:
            if len(self.goal_cell_locs) == 0:
                print("Running value iteration with no goals assigned..")
            policy = self.run_value_iteration().policy

        for init_state in traj_init_states:
            action_seq, state_seq = self.plan(init_state, policy=policy,
                                              horizon=horizon)
            traj_states_list.append(state_seq)
            traj_action_list.append(action_seq)

        return traj_states_list, traj_action_list

    # ---------------------
    # -- Value Iteration --
    # ---------------------
    def run_value_iteration(self):
        """Runs value iteration (if needed).

        Returns:
            ValueIteration object.
        """
        # If value iteration was run previously, don't re-run it
        if self._policy_invalidated == True:
            self.value_iter = ValueIteration(self, sample_rate=1)
            _ = self.value_iter.run_vi()
            self._policy_invalidated = False
        return self.value_iter

    def get_value_grid(self):
        """Returns value over states space grid.

        """
        value_iter = self.run_value_iteration()
        V = np.zeros((self.height, self.width), dtype=np.float32)
        for row in range(self.height):
            for col in range(self.width):
                x, y = self._rowcol_to_xy(row, col)
                V[row, col] = value_iter.value_func[NavigationWorldState(x, y)]
        return V

    def get_all_states(self):
        """Returns all states.

        """
        return [NavigationWorldState(x, y) for x in range(1, self.width + 1)
                for y in range(1, self.height + 1)]

    def get_reachable_states(self):
        """Returns all reachable states from @self.init_loc.

        """
        return self.value_iter.get_states()

    def get_trans_dict(self):
        """Returns transition dynamics matrix.

        """
        self.value_iter._compute_matrix_from_trans_func()
        return self.value_iter.trans_dict

    # --------------
    # -- Features --
    # --------------
    def __transfrom(self, mat, type):
        if type == "normalize_manhattan":
            return mat / (self.width + self.height)
        if type == "normalize_euclidean":
            return mat / np.sqrt(self.width**2 + self.height**2)
        else:
            return mat

    def compute_grid_distance_features(self, incl_cells, incl_goals,
                                       normalize=False):
        """Computes distances to specified cell types for entire grid.

        Returns:
            3D array (row, col, distance)
        """
        # Convert 2 flags to decimal representation. This is useful to check if
        # requested features are different from those previously stored
        feature_kind = int(str(int(incl_cells)) + str(int(incl_goals)), 2)
        if self.feature_cell_dist is not None \
                and self.feature_cell_dist_kind == feature_kind:
            return self.__transfrom(self.feature_cell_dist,
                                    "normalize_manhattan" if normalize
                                    else "None")

        dist_cell_ids = copy.deepcopy(
            self.nav_cell_ids) if incl_cells else []
        dist_cell_ids += self.goal_cell_ids if incl_goals else []
        loc_cells = [
            np.vstack(np.where(self.map_state_cell_id == cell_id)).transpose()
            for cell_id in dist_cell_ids]

        self.feature_cell_dist = np.zeros(
            self.map_state_cell_id.shape + (len(dist_cell_ids),), np.float32)

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

    def cell_id_ind_feature(self, cell_id, include_goal=True):
        """Indicator feature for cell_id.

        """
        if include_goal:
            return np.eye(len(self.combined_cell_ids))[cell_id]
        else:
            # use 0 vector for goals
            return np.vstack(
                (np.eye(len(self.nav_cell_ids)),
                 np.zeros((len(self.wall_cell_ids), len(self.nav_cell_ids))),
                 np.zeros((len(self.goal_cell_ids), len(self.nav_cell_ids))))
            )[cell_id]

    def feature_at_loc(self, x, y, feature_type="indicator",
                       incl_cell_distances=False, incl_goal_indicator=True,
                       incl_goal_distances=False, normalize_distance=False,
                       dtype=np.float32):
        """Returns feature vector at a state corresponding to (x, y) location.

        Args:
            x, y (int, int): cartesian coordinates in 2d state space
            feature_type (str): "indicator" to use one-hot encoding of
                cell type "cartesian" to use (x, y) and "rowcol" to use
                (row, col) as feature.
            incl_cell_distances (bool): Include distance to each type of cell.
            incl_goal_distances (bool): Include distance to the goals.
            incl_goal_indicator: True - adds goal indicator feature
                If all goals have same color, it'll add single indicator
                variable for all goals, otherwise it'll use different
                indicator variables for each goal.
                (only applicable for "indicator" feature type).
            normalize_distance (bool): Whether to normalize cell type
                distances to 0-1 range (only used when
                "incl_distance_features" is True).
            dtype (numpy datatype): cast feature vector to dtype
        """
        row, col = self._xy_to_rowcol(x, y)
        assert feature_type in ["indicator", "cartesian", "rowcol"]

        if feature_type == "indicator":
            ind_feature = self.cell_id_ind_feature(
                self.map_state_cell_id[row, col], incl_goal_indicator)
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
                         incl_cell_distances=False, incl_goal_indicator=True,
                         incl_goal_distances=False, normalize_distance=False,
                         dtype=np.float32):
        """Returns feature vector at a state corresponding to MdP State.

        Args:
            mdp_state (int, int): GridWorldState object
            feature_type (str): "indicator" to use one-hot encoding of
                cell type "cartesian" to use (x, y) and "rowcol" to use
                (row, col) as feature.
            incl_cell_distances (bool): Include distance to each type of cell.
            incl_goal_distances (bool): Include distance to the goals.
            incl_goal_indicator: True - adds goal indicator feature
                If all goals have same color, it'll add single indicator
                variable for all goals, otherwise it'll use different
                indicator variables for each goal.
                (only applicable for "indicator" feature type).
            normalize_distance (bool): Whether to normalize cell type
                distances to 0-1 range (only used when
                "incl_distance_features" is True).
            dtype (numpy datatype): cast feature vector to dtype
        """
        return self.feature_at_loc(mdp_state.x, mdp_state.y, feature_type,
                                   incl_cell_distances, incl_goal_indicator,
                                   incl_goal_distances, normalize_distance,
                                   dtype)

    # -------------------
    # -- Visualization --
    # -------------------
    @staticmethod
    def __display_text(ax, x_start, x_end, y_start, y_end, values,
                       fontsize=12):
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
                       axis_tick_font_sz=8, title=None):
        """Visualize Navigation World.

        Args:
            values (2d ndarray): Values to be visualized in the grid,
                defaults to cell types.
            cmap (Matplotlib Colormap): Colormap corresponding to values,
                defaults to ListedColormap with colors specified in
                @self.nav_cell_types and @self.goal_cell_types
            trajectories: Trajectories to be shown on the grid.
            subplot_str (str): Subplot number string (e.g., "411", "412", etc.)
            new_fig (bool): Whether to use existing figure context.
            show_rewards_colorbar (bool): Shows colorbar with cell
                reward values.
            title (str): Title of the plot.
        """
        if new_fig:
            plt.figure(
                figsize=(max(self.height // 4, 6), max(self.width // 4, 6)))
        # Subplot if needed
        if subplot_str is not None:
            plt.subplot(subplot_str)

        # Use state space (cell types) if values is None
        if values is None:
            values = self.map_state_cell_id.copy()

        # Colormap
        if cmap is not None and state_space_cmap:
            norm = colors.Normalize(vmin=0,
                                    vmax=len(self.combined_cell_types)-1)
            # Leave string colors as it is, convert int colors to
            # normalized rgba
            cell_colors = [
                cmap(norm(cell)) if isinstance(cell, int) else cell
                for cell in self.combined_cell_types]
            cmap = colors.ListedColormap(cell_colors, N=self.n_unique_cells)

        # Plot values
        im = plt.imshow(values, interpolation='None', cmap=cmap)
        plt.title(title if title else self.name)
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
                plt.plot(path_xs, path_ys, traj_marker,
                         linewidth=traj_linewidth)
                plt.plot(path_xs[0], path_ys[0], init_marker,
                         markersize=init_marker_sz)  # Mark init state
                plt.plot(path_xs[-1], path_ys[-1], end_marker,
                         markersize=end_marker_sz)  # Mark end state
        # Mark goals
        if len(self.goal_cell_locs) != 0:
            for goal_cells in self.goal_cell_locs:
                for goal_x, goal_y in goal_cells:
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
                cb = plt.colorbar(im, ticks=range(len(self.cell_type_rewards)),
                                  cax=cax)
                cb.set_ticklabels(self.cell_type_rewards)
            else:
                plt.colorbar(im, cax=cax)
        if subplot_str is None:
            plt.show()
