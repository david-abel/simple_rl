# Python imports.
from collections import defaultdict
import random

# Other imports.
from simple_rl.mdp.StateClass import State

class Option(object):

	def __init__(self, init_predicate, term_predicate, policy, name="o", term_prob=0.00):
		'''
		Args:
			init_func (S --> {0,1})
			init_func (S --> {0,1})
			policy (S --> A)
		'''
		self.init_predicate = init_predicate
		self.term_predicate = term_predicate
		self.term_flag = False
		self.name = name
		self.term_prob = term_prob

		if type(policy) is defaultdict or type(policy) is dict:
			self.policy_dict = dict(policy)
			self.policy = self.policy_from_dict
		else:
			self.policy = policy

	def is_init_true(self, ground_state):
		return self.init_predicate.is_true(ground_state)

	def is_term_true(self, ground_state):
		return self.term_predicate.is_true(ground_state) or self.term_flag or self.term_prob > random.random()

	def act(self, ground_state):
		return self.policy(ground_state)

	def set_policy(self, policy):
		self.policy = policy

	def set_name(self, new_name):
		self.name = new_name

	def act_until_terminal(self, cur_state, transition_func):
		'''
		Summary:
			Executes the option until termination.
		'''
		if self.is_init_true(cur_state):
			cur_state = transition_func(cur_state, self.act(cur_state))
			while not self.is_term_true(cur_state):
				cur_state = transition_func(cur_state, self.act(cur_state))

		return cur_state

	def rollout(self, cur_state, reward_func, transition_func, max_rollout_depth, step_cost=0):
		'''
		Summary:
			Executes the option until termination.

		Args:
			cur_state (simple_rl.State)
			reward_func (lambda)
			transition_func (lambda)
			max_rollout_depth (int)
			step_cost (float)

		Returns:
			(tuple):
				1. (State): state we landed in.
				2. (float): Reward from the trajectory.
		'''
		total_reward = 0
		rollout_depth = 0

		if self.is_init_true(cur_state):
			# First step.
			total_reward += reward_func(cur_state, self.act(cur_state)) - step_cost
			cur_state = transition_func(cur_state, self.act(cur_state))
			# Act until terminal.
			while not self.is_term_true(cur_state) and not cur_state.is_terminal():
				total_reward += reward_func(cur_state, self.act(cur_state)) - step_cost
				cur_state = transition_func(cur_state, self.act(cur_state))
				rollout_depth += 1

		return cur_state, total_reward

	def policy_from_dict(self, state):
		if state not in self.policy_dict.keys():
			self.term_flag = True
			return random.choice(list(set(self.policy_dict.values())))
		else:
			self.term_flag = False
			return self.policy_dict[state]

	def term_func_from_list(self, state):
		return state in self.term_list

	def __str__(self):
		return "option." + str(self.name)