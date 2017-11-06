# Python imports.
import random

# Other imports.
from PolicyClass import Policy
from collections import defaultdict

class PolicyFromDict(Policy):

	def __init__(self, policy_dict={}):
		self.policy_dict = policy_dict

	def get_action(self, state):
		# if state not in self.policy_dict.keys():
		# 	print "(PolicyFromDict) Abstraction Error:", state, "never seen before."
		# 	quit()

		if state not in self.policy_dict.keys():
			# print "(PolicyFromDict) Warning: unseen state (" + str(state) + "). Acting randomly."
			return random.choice(list(set(self.policy_dict.values())))
		else:
			# print "Seen state!:" + str(state)
			return self.policy_dict[state]
