# Python imports.
import random
import Queue
from collections import defaultdict

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.abstraction.action_abs import ActionAbstraction
from simple_rl.abstraction.state_abs import StateAbstraction
from simple_rl.abstraction.dev_hierarch.make_abstr_mdp import make_abstr_mdp
from simple_rl.planning.PlannerClass import Planner
from simple_rl.planning.ValueIterationClass import ValueIteration

class AbstractValueIteration(ValueIteration):
    ''' AbstractValueIteration: Runs ValueIteration on an abstract MDP induced by the given state and action abstraction '''

    def __init__(self, ground_mdp, state_abstr=None, action_abstr=None, sample_rate=10, delta=0.001, max_iterations=1000):
        '''
        Args:
            ground_mdp (simple_rl.MDP)
            state_abstr (simple_rl.StateAbstraction)
            action_abstr (simple_rl.ActionAbstraction)
        '''
        self.ground_mdp = ground_mdp

        # If None is given for either, set the sa/aa to defaults.
        self.state_abstr = state_abstr if state_abstr is not None else StateAbstraction()
        self.action_abstr = action_abstr if action_abstr is not None else ActionAbstraction(prim_actions=ground_mdp.get_actions())

        mdp = make_abstr_mdp(ground_mdp, self.state_abstr, self.action_abstr, step_cost=0.0)

        ValueIteration.__init__(self, mdp, sample_rate, delta, max_iterations)

    def policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): Action

        Summary:
            For use in a FixedPolicyAgent.

        # TODO:
            Doesn't account for options terminating (policy is over options, currently just grounds them).
        '''
        option = self._get_max_q_action(self.state_abstr.phi(state))
        return option.act(state)
