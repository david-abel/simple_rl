# Python imports.
from collections import defaultdict
import copy

# Other imports.
from simple_rl.planning import Planner
from simple_rl.planning import ValueIteration
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning.BoundedRTDPClass import BoundedRTDP

class MonotoneLowerBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        relaxed_mdp = MonotoneLowerBound._construct_deterministic_relaxation_mdp(mdp)

        Planner.__init__(self, relaxed_mdp, name)
        self.vi = ValueIteration(relaxed_mdp)
        self.states = self.vi.get_states()
        self.vi._compute_matrix_from_trans_func()
        self.vi.run_vi()
        self.lower_values = self._construct_lower_values()

    @staticmethod
    def _construct_deterministic_relaxation_mdp(mdp):
        relaxed_mdp = copy.deepcopy(mdp)
        relaxed_mdp.set_slip_prob(0.0)
        return relaxed_mdp

    def _construct_lower_values(self):
        values = defaultdict()
        for state in self.states:
            values[state] = self.vi.get_value(state)
        return values

class MonotoneUpperBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        Planner.__init__(self, mdp, name)
        self.vi = ValueIteration(mdp)
        self.states = self.vi.get_states()
        self.upper_values = self._construct_upper_values()

    def _construct_upper_values(self):
        values = defaultdict()
        for state in self.states:
            values[state] = 1. / (1. - self.gamma)
        return values

def main():
    test_mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.2)
    lower_value_function = MonotoneLowerBound(test_mdp).lower_values
    upper_value_function = MonotoneUpperBound(test_mdp).upper_values
    bounded_rtdp = BoundedRTDP(test_mdp, lower_values_init=lower_value_function, upper_values_init=upper_value_function)
    test_policy = bounded_rtdp.plan()
    print('Derived policy:\n{}'.format(test_policy))

if __name__ == '__main__':
    main()
