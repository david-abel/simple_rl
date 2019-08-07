'''
simple_rl
	abstraction/
		action_abs/
		state_abs/
		...
	agents/
		AgentClass.py
		QLearningAgentClass.py
		RandomAgentClass.py
		RMaxAgentClass.py
		...
	experiments/
		ExperimentClass.py
		ExperimentParameters.py
	mdp/
		MDPClass.py
		StateClass.py
	planning/
		BeliefSparseSamplingClass.py
		MCTSClass.py
		PlannerClass.py
		ValueIterationClass.py
	pomdp/
		BeliefMDPClass.py
		BeliefStateClass.py
		BeliefUpdaterClass.py
		POMDPClass.py
	tasks/
		chain/
			ChainMDPClass.py
			ChainStateClass.py
		grid_world/
			GridWorldMPDClass.py
			GridWorldStateClass.py
		...
	utils/
		chart_utils.py
		make_mdp.py
	run_experiments.py

Author and Maintainer: David Abel (david_abel.github.io)
Last Updated: March 27th, 2019
Contact: david_abel@brown.edu
License: Apache
'''
# Fix xrange to cooperate with python 2 and 3.
try:
    xrange
except NameError:
    xrange = range

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

# Imports.
import simple_rl.abstraction, simple_rl.agents, simple_rl.experiments, simple_rl.mdp, simple_rl.planning, simple_rl.tasks, simple_rl.utils
import simple_rl.run_experiments

from simple_rl._version import __version__