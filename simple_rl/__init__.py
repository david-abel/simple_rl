'''
simple_rl
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

Author and Maintainer: David Abel (cs.brown.edu/~dabel/)
Last Updated: April 13th, 2018
Contact: dabel@cs.brown.edu
License: Apache
'''
# Fix xrange to cooperate with python 2 and 3.
try:
    xrange
except NameError:
    xrange = range

# Imports.
import simple_rl.agents, simple_rl.experiments, simple_rl.mdp, simple_rl.planning, simple_rl.tasks, simple_rl.utils
import simple_rl.run_experiments

from simple_rl._version import __version__