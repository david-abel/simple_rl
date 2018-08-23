# Python imports.
from collections import defaultdict
import numpy as np

# Other imports.
from simple_rl.planning import ValueIteration
from simple_rl.mdp import MDP
from simple_rl.mdp import MDPDistribution
from simple_rl.abstraction.state_action_abstr_mdp.RewardFuncClass import RewardFunc
from simple_rl.abstraction.state_action_abstr_mdp.TransitionFuncClass import TransitionFunc
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction
# ------------------
# -- Single Level --
# ------------------

def make_abstr_mdp(mdp, state_abstr, action_abstr=None, step_cost=0.0, sample_rate=5):
	'''
	Args:
		mdp (MDP)
		state_abstr (StateAbstraction)
		action_abstr (ActionAbstraction)
		step_cost (float): Cost for a step in the lower MDP.
		sample_rate (int): Sample rate for computing the abstract R and T.

	Returns:
		(MDP)
	'''

	if action_abstr is None:
		action_abstr = ActionAbstraction(prim_actions=mdp.get_actions())

	# Make abstract reward and transition functions.
	def abstr_reward_lambda(abstr_state, abstr_action):
		if abstr_state.is_terminal():
			return 0

		# Get relevant MDP components from the lower MDP.
		lower_states = state_abstr.get_lower_states_in_abs_state(abstr_state)
		lower_reward_func = mdp.get_reward_func()
		lower_trans_func = mdp.get_transition_func()

		# Compute reward.
		total_reward = 0
		for ground_s in lower_states:
			for sample in range(sample_rate):
				s_prime, reward = abstr_action.rollout(ground_s, lower_reward_func, lower_trans_func, step_cost=step_cost)
				total_reward += float(reward) / (len(lower_states) * sample_rate) # Add weighted reward.

		return total_reward

	def abstr_transition_lambda(abstr_state, abstr_action):
		is_ground_terminal = False
		for s_g in state_abstr.get_lower_states_in_abs_state(abstr_state):
			if s_g.is_terminal():
				is_ground_terminal = True
				break

		# Get relevant MDP components from the lower MDP.
		if abstr_state.is_terminal():
			return abstr_state

		lower_states = state_abstr.get_lower_states_in_abs_state(abstr_state)
		lower_reward_func = mdp.get_reward_func()
		lower_trans_func = mdp.get_transition_func()


		# Compute next state distribution.
		s_prime_prob_dict = defaultdict(int)
		total_reward = 0
		for ground_s in lower_states:
			for sample in range(sample_rate):
				s_prime, reward = abstr_action.rollout(ground_s, lower_reward_func, lower_trans_func)
				s_prime_prob_dict[s_prime] += (1.0 / (len(lower_states) * sample_rate)) # Weighted average.
		
		# Form distribution and sample s_prime.
		next_state_sample_list = list(np.random.multinomial(1, list(s_prime_prob_dict.values())).tolist())
		end_ground_state = list(s_prime_prob_dict.keys())[next_state_sample_list.index(1)]
		end_abstr_state = state_abstr.phi(end_ground_state)

		return end_abstr_state
	
	# Make the components of the Abstract MDP.
	abstr_init_state = state_abstr.phi(mdp.get_init_state())
	abstr_action_space = action_abstr.get_actions()
	abstr_state_space = state_abstr.get_abs_states()
	abstr_reward_func = RewardFunc(abstr_reward_lambda, abstr_state_space, abstr_action_space)
	abstr_transition_func = TransitionFunc(abstr_transition_lambda, abstr_state_space, abstr_action_space, sample_rate=sample_rate)

	# Make the MDP.
	abstr_mdp = MDP(actions=abstr_action_space,
                    init_state=abstr_init_state,
                    reward_func=abstr_reward_func.reward_func,
                    transition_func=abstr_transition_func.transition_func,
                    gamma=mdp.get_gamma())

	return abstr_mdp

def make_abstr_mdp_distr(mdp_distr, state_abstr, action_abstr, step_cost=0.1):
	'''
	Args:
		mdp_distr (MDPDistribution)
		state_abstr (StateAbstraction)
		action_abstr (ActionAbstraction)

	Returns:
		(MDPDistribution)
	'''

	# Loop through old mdps and abstract.
	mdp_distr_dict = {}
	for mdp in mdp_distr.get_all_mdps():
		abstr_mdp = make_abstr_mdp(mdp, state_abstr, action_abstr, step_cost=step_cost)
		prob_of_abstr_mdp = mdp_distr.get_prob_of_mdp(mdp)
		mdp_distr_dict[abstr_mdp] = prob_of_abstr_mdp

	return MDPDistribution(mdp_distr_dict)

# -----------------
# -- Multi Level --
# -----------------

def make_abstr_mdp_multi_level(mdp, state_abstr_stack, action_abstr_stack, step_cost=0.1, sample_rate=5):
	'''
	Args:
		mdp (MDP)
		state_abstr_stack (StateAbstractionStack)
		action_abstr_stack (ActionAbstractionStack)
		step_cost (float): Cost for a step in the lower MDP.
		sample_rate (int): Sample rate for computing the abstract R and T.

	Returns:
		(MDP)
	'''
	mdp_level = min(state_abstr_stack.get_num_levels(), action_abstr_stack.get_num_levels())

	for i in range(1, mdp_level + 1):
		state_abstr_stack.set_level(i)
		action_abstr_stack.set_level(i)
		mdp = make_abstr_mdp(mdp, state_abstr_stack, action_abstr_stack, step_cost, sample_rate)

	return mdp

def make_abstr_mdp_distr_multi_level(mdp_distr, state_abstr, action_abstr, step_cost=0.1):
	'''
	Args:
		mdp_distr (MDPDistribution)
		state_abstr (StateAbstraction)
		action_abstr (ActionAbstraction)

	Returns:
		(MDPDistribution)
	'''

	# Loop through old mdps and abstract.
	mdp_distr_dict = {}
	for mdp in mdp_distr.get_all_mdps():
		abstr_mdp = make_abstr_mdp_multi_level(mdp, state_abstr, action_abstr, step_cost=step_cost)
		prob_of_abstr_mdp = mdp_distr.get_prob_of_mdp(mdp)
		mdp_distr_dict[abstr_mdp] = prob_of_abstr_mdp

	return MDPDistribution(mdp_distr_dict)

def _rew_dict_from_lambda(input_lambda, state_space, action_space, sample_rate):
	result_dict = defaultdict(lambda:defaultdict(float))
	for s in state_space:
		for a in action_space:
			for i in range(sample_rate):
				result_dict[s][a] = input_lambda(s,a) / sample_rate

	return result_dict

