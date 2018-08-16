''' BoundedRTDPClass.py: Contains the Bounded-RTPDP solver class. '''

# Python imports.
from collections import defaultdict
import copy

# Other imports.
from simple_rl.planning import Planner, ValueIteration
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State
from simple_rl.utils.additional_datastructures import SimpleRLStack

class BoundedRTDP(Planner):
    '''
    Bounded Real-Time Dynamic Programming: RTDP with monotone upper bounds and performance guarantees (McMahan et al)

    The Bounded RTDP solver can produce partial policies with strong performance guarantees while only touching a
    fraction of the state space, even on problems where other algorithms would have to visit the full state space.
    To do so, Bounded RTDP maintains both upper and lower bounds on the optimal value function.
    '''
    def __init__(self, mdp, lower_values_init, upper_values_init, tau=10., name='BRTDP'):
        '''
        Args:
            mdp (MDP): underlying MDP to plan in
            lower_values_init (defaultdict): lower bound initialization on the value function
            upper_values_init (defaultdict): upper bound initialization on the value function
            tau (float): scaling factor to help determine when the bounds on the value function are tight enough
            name (str): Name of the planner
        '''
        Planner.__init__(self, mdp, name)
        self.lower_values = lower_values_init
        self.upper_values = upper_values_init

        # Using the value iteration class for accessing the matrix of transition probabilities
        vi = ValueIteration(mdp, sample_rate=1000)
        self.states = vi.get_states()
        vi._compute_matrix_from_trans_func()
        self.trans_dict = vi.trans_dict

        self.max_diff = (self.upper_values[self.mdp.init_state] - self.lower_values[self.mdp.init_state]) / tau

    # ------------------
    # -- Planning API --
    # ------------------

    def plan(self, state=None, horizon=100):
        '''
        Main function of the Planner class.
        Args:
            state (State)
            horizon (int)

        Returns:
            policy (defaultdict)
        '''

        # Run the BRTDP algorithm to perform value function rollouts
        self.run_sample_trial()

        # Continue planning based on the heuristic refinement performed above
        state = self.mdp.get_init_state() if state is None else state
        policy = defaultdict()
        steps = 0
        while (not state.is_terminal()) and steps < horizon:
            next_action = self.policy(state)
            policy[state] = next_action
            state = self.transition_func(state, next_action)
            steps += 1
        return policy

    def policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            action (str)
        '''
        return self._greedy_action(state, self.lower_values)

    # ------------------
    # -- RTDP Routine --
    # ------------------

    def run_sample_trial(self, verbose=False):
        init_state = self.mdp.init_state
        state = copy.deepcopy(init_state)
        trajectory = SimpleRLStack()
        while not state.is_terminal():
            trajectory.push(state)
            self.upper_values[state] = self._best_qvalue(state, self.upper_values)
            action = self._greedy_action(state, self.lower_values)
            self.lower_values[state] = self._qvalue(state, action, self.lower_values)
            expected_gap_distribution = self._expected_gap_distribution(state, action)
            expected_gap = sum(expected_gap_distribution.values())
            if verbose: print('{}\tAction: {}\tGap: {}\tMaxDiff: {}'.format(state, action, expected_gap, self.max_diff))
            if expected_gap < self.max_diff:
                if verbose: print('Ending rollouts with gap {} and max_diff {}'.format(expected_gap, self.max_diff))
                break
            state = BoundedRTDP._pick_next_state(expected_gap_distribution, expected_gap)
        while not trajectory.is_empty():
            state = trajectory.pop()
            self.upper_values[state] = self._best_qvalue(state, self.upper_values)
            self.lower_values[state] = self._best_qvalue(state, self.lower_values)

    def _greedy_action(self, state, values):
        return max([(self._qvalue(state, action, values), action) for action in self.actions])[1]

    def _qvalue(self, state, action, values):
        return self.mdp.reward_func(state, action) + sum([self.trans_dict[state][action][next_state] * values[next_state] \
                                                      for next_state in self.states])

    def _best_qvalue(self, state, values):
        return max([self._qvalue(state, action, values) for action in self.actions])

    # -------------------------
    # -- Convenience Methods --
    # -------------------------

    def _expected_gap_distribution(self, state, action):
        '''
        Weight the distribution representing our uncertainty over state values by the
        transition probabilities when taking `action` from `state`.
        Args:
            state (State)
            action (str)

        Returns:
            expected_gaps (defaultdict): weighted distribution over states of difference b/w upper and lower values
        '''
        expected_gaps = defaultdict()
        for next_state in self.states:
            gap = self.upper_values[next_state] - self.lower_values[next_state]
            expected_gaps[next_state] = self.trans_dict[state][action][next_state] * gap
        return expected_gaps

    @staticmethod
    def _pick_next_state(distribution, expected_gap):
        '''
        Args:
            distribution (defaultdict)
            expected_gap (float)

        Returns:
            state (State): sampled from a scaled version of `distribution`
        '''
        def _scale_distribution(_distribution, scaling_factor):
            for state in _distribution:
                _distribution[state] *= scaling_factor
            return _distribution
        scaled_distribution = _scale_distribution(distribution, expected_gap)
        return max(scaled_distribution, key=scaled_distribution.get)
