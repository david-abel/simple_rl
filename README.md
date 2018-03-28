# simple_rl
A simple framework for experimenting with Reinforcement Learning in Python.

There are loads of other great libraries out there for RL. The aim of this one is twofold:

1. Simplicity.
2. Reproducibility of results.

A brief tutorial for a slightly earlier version is available [here](http://cs.brown.edu/~dabel/blog/posts/simple_rl.html). As of version 0.77, the library should work with both Python 2 and Python 3. Please let me know if you find that is not the case!

simple_rl requires [numpy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/). Some MDPs have visuals, too, which requires [pygame](http://www.pygame.org/news). Also includes support for hooking into any of the [Open AI Gym environments](https://gym.openai.com/envs). I recently added a basic test script, contained in the _tests_ directory.


## Installation

The easiest way to install is with [pip](https://pypi.python.org/pypi/pip). Just run:

	pip install simple_rl

Alternatively, you can download simple_rl [here](https://github.com/david-abel/simple_rl/tarball/v0.76).

## Example

Some examples showcasing basic functionality are included in the _examples_ directory.

To run a simple experiment, import the _run_agents_on_mdp(agent_list, mdp)_ method from _simple_rl.run_experiments_ and call it with some agents for a given MDP. For example:

	# Imports
	from simple_rl.run_experiments import run_agents_on_mdp
	from simple_rl.tasks import GridWorldMDP
	from simple_rl.agents import QLearningAgent

	# Run Experiment
	mdp = GridWorldMDP()
	agent = QLearningAgent(mdp.get_actions())
	run_agents_on_mdp([agent], mdp)

## Overview

* (_agents_): Code for some basic agents (a random actor, _Q_-learner, [[R-Max]](http://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf), _Q_-learner with a Linear Approximator, and so on).

* (_experiments_): Code for an Experiment class to track parameters and reproduce results.

* (_mdp_): Code for a basic MDP and MDPState class, and an MDPDistribution class (for  lifelong learning). Also contains OO-MDP implementation [[Diuk et al. 2008]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.7056&rep=rep1&type=pdf).

* (_planning_): Implementations for planning algorithms, includes ValueIteration and MCTS [[Couloum 2006]](https://hal.archives-ouvertes.fr/file/index/docid/116992/filename/CG2006.pdf), the latter being still in development.

* (_tasks_): Implementations for a few standard MDPs (grid world, N-chain, Taxi [[Dietterich 2000]](http://www.scs.cmu.edu/afs/cs/project/jair/pub/volume13/dietterich00a.pdf), and the [OpenAI Gym](https://gym.openai.com/envs)).

* (_utils_): Code for charting and other utilities.


## Making a New MDP

Make an MDP subclass, which needs:

* A static variable, _ACTIONS_, which is a list of strings denoting each action.

* Implement a reward and transition function and pass them to MDP constructor (along with _ACTIONS_).

* I also suggest overwriting the "\_\_str\_\_" method of the class, and adding a "\_\_init\_\_.py" file to the directory.

* Create a State subclass for your MDP (if necessary). I suggest overwriting the "\_\_hash\_\_", "\_\_eq\_\_", and "\_\_str\_\_" for the class to play along well with the agents.


## Making a New Agent

Make an Agent subclass, which requires:

* A method, _act(self, state, reward)_, that returns an action.

* A method, _reset()_, that puts the agent back to its _tabula rasa_ state.

## In Development

A few features are in development, including MCTS [[Coloum 2006]](https://hal.inria.fr/file/index/docid/116992/filename/CG2006.pdf) and the DQN [[Mnih et al. 2015]](http://www.davidqiu.com:8888/research/nature14236.pdf).

Let me know if you have any questions or suggestions.

Cheers,

-Dave