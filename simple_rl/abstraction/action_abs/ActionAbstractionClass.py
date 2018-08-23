# Python imports.
from __future__ import print_function
from collections import defaultdict
import random

# Other imports.
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class ActionAbstraction(object):

    def __init__(self, options=None, prim_actions=[], term_prob=0.0, prims_on_failure=False):
        self.options = options if options is not None else self._convert_to_options(prim_actions)
        self.is_cur_executing = False
        self.cur_option = None # The option we're executing currently.
        self.term_prob = term_prob
        self.prims_on_failure = prims_on_failure
        if self.prims_on_failure:
            self.prim_actions = prim_actions

    def act(self, agent, abstr_state, ground_state, reward):
        '''
        Args:
            agent (Agent)
            abstr_state (State)
            ground_state (State)
            reward (float)

        Returns:
            (str)
        '''

        if self.is_next_step_continuing_option(ground_state) and random.random() > self.term_prob:
            # We're in an option and not terminating.
            return self.get_next_ground_action(ground_state)
        else:
            # We're not in an option, check with agent.
            active_options = self.get_active_options(ground_state)

            if len(active_options) == 0:
                if self.prims_on_failure:
                    # In a failure state, back off to primitives.
                    agent.actions = self._convert_to_options(self.prim_actions)
                else:
                    # No actions available.
                    raise ValueError("(simple_rl) Error: no actions available in state " + str(ground_state) + ".")
            else:
                # Give agent available options.
                agent.actions = active_options
            
            abstr_action = agent.act(abstr_state, reward)
            self.set_option_executing(abstr_action)

            return self.abs_to_ground(ground_state, abstr_action)

    def get_active_options(self, state):
        '''
        Args:
            state (State)

        Returns:
            (list): Contains all active options.
        '''

        return [o for o in self.options if o.is_init_true(state)]

    def _convert_to_options(self, action_list):
        '''
        Args:
            action_list (list)

        Returns:
            (list of Option)
        '''
        options = []
        for ground_action in action_list:
            o = ground_action
            if type(ground_action) is str:
                o = Option(init_predicate=Predicate(make_lambda(True)),
                            term_predicate=Predicate(make_lambda(True)),
                            policy=make_lambda(ground_action),
                            name="prim." + ground_action)
            else:
                print(type(ground_action))
            options.append(o)
        return options

    def is_next_step_continuing_option(self, ground_state):
        '''
        Returns:
            (bool): True iff an option was executing and should continue next step.
        '''
        return self.is_cur_executing and not self.cur_option.is_term_true(ground_state)

    def set_option_executing(self, option):
        if option not in self.options and "prim" not in option.name:
            raise ValueError("(simple_rl) Error: agent chose a non-existent option (" + str(option) + ").")

        self.cur_option = option
        self.is_cur_executing = True

    def get_next_ground_action(self, ground_state):
        return self.cur_option.act(ground_state)

    def get_actions(self):
        return list(self.options)

    def abs_to_ground(self, ground_state, abstr_action):
        return abstr_action.act(ground_state)

    def add_option(self, option):
        self.options += [option]

    def reset(self):
        self.is_cur_executing = False
        self.cur_option = None # The option we're executing currently.

    def end_of_episode(self):
        self.reset()


def make_lambda(result):
    return lambda x : result
