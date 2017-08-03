'''
simple_rl
	agents/
		AgentClass.py
		QLearnerAgentClass.py
		RandomAgentClass.py
		RMaxAgentClass.py
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
	utils/
		chart_utils.py
	run_experiments.py

Author and Maintainer: David Abel (cs.brown.edu/~dabel/)
Last Updated: July 25th, 2017
Contact: dabel@cs.brown.edu
License: MIT
'''
import agents, experiments, mdp, planning, tasks, utils
import run_experiments