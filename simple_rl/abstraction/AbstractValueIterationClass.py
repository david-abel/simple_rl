# Python imports.
import random
from collections import defaultdict

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.state_action_abstr_mdp import abstr_mdp_funcs
from simple_rl.planning.PlannerClass import Planner
from simple_rl.planning.ValueIterationClass import ValueIteration

class AbstractValueIteration(ValueIteration):
    ''' AbstractValueIteration: Runs ValueIteration on an abstract MDP induced by the given state and action abstraction '''

    def __init__(self, ground_mdp, state_abstr=None, action_abstr=None, vi_sample_rate=5, max_iterations=1000, amdp_sample_rate=5, delta=0.001):
        '''
        Args:
            ground_mdp (simple_rl.MDP)
            state_abstr (simple_rl.StateAbstraction)
            action_abstr (simple_rl.ActionAbstraction)
            vi_sample_rate (int): Num samples per transition for running VI.
            max_iterations (int): Usual VI # Iteration bound.
            amdp_sample_rate (int): Num samples per abstract transition to use for computing R_abstract, T_abstract.
        '''
        self.ground_mdp = ground_mdp
    
        # Grab ground state space.
        vi = ValueIteration(self.ground_mdp, delta=0.001, max_iterations=1000, sample_rate=5)
        state_space = vi.get_states()

        # Make the abstract MDP.
        self.state_abstr = state_abstr if state_abstr is not None else StateAbstraction(ground_state_space=state_space)
        self.action_abstr = action_abstr if action_abstr is not None else ActionAbstraction(prim_actions=ground_mdp.get_actions())
        abstr_mdp = abstr_mdp_funcs.make_abstr_mdp(ground_mdp, self.state_abstr, self.action_abstr, step_cost=0.0, sample_rate=amdp_sample_rate)

        # Create VI with the abstract MDP.
        ValueIteration.__init__(self, abstr_mdp, vi_sample_rate, delta, max_iterations)

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
