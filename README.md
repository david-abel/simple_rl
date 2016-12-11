# simple_rl
A simple framework for experimenting with Reinforcement Learning in Python 2.7 (brief tutorial [here](http://cs.brown.edu/~dabel/blog/posts/simple_rl.html)).

There are loads of other great libraries out there for RL. The aim of this one is twofold:

1. Simplicity.
2. Reproducibility of results.

See _run_experiments.py_ to run some basic experiments.

Just requires [numpy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/). If you want to use the [Gradient Booster](http://arxiv.org/abs/1603.04119) agent, you'll need [sklearn](http://scikit-learn.org/stable/).

## Installation

The easiest way to install is with [pip](https://pypi.python.org/pypi/pip). Just run:

	pip install simple_rl

Alternatively, you can download simple_rl [here](https://github.com/david-abel/simple_rl/tarball/v0.6).

## Example

To run a simple experiment, import the _run_agents_on_mdp(agent_list, mdp)_ method from _simple_rl.run_experiments_ and call it with some agents for a given MDP. For example:

	# Imports
	from simple_rl.run_experiments import run_agents_on_mdp
	from simple_rl.tasks import GridWorldMDP
	from simple_rl.agents import QLearnerAgent

	# Run Experiment
	mdp = GridWorldMDP(10, 10, (1, 1), (10, 10))
	agent = QLearnerAgent(mdp.actions)
	run_agents_on_mdp([agent], mdp)

## Overview

* (_agents_): Code for some basic agents (a random actor, _Q_-learner, [[R-Max]](http://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf), _Q_-learner with a Linear Approximator, etc.).

* (_experiments_): Code for an Experiment class to reproduce results.

* (_mdp_): Code for a basic MDP and MDPState class. Also contains OO-MDP implementation [[Diuk et al. 2008]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.7056&rep=rep1&type=pdf).

* (_tasks_): Implementations for a few standard MDPs (grid world, n-chain, and Taxi [[Dietterich 2000]](http://www.scs.cmu.edu/afs/cs/project/jair/pub/volume13/dietterich00a.pdf)). Recently added support for the [Arcade Learning Enviroment](http://www.arcadelearningenvironment.org/): to run experiments with ALE you'll also need at least one [Atari 2600 ROM](http://www.atariage.com/system_items.html?SystemID=2600&ItemTypeID=ROM).

* (_utils_): Code for charting utilities.


## Making a New MDP

Make an MDP subclass, which needs:

* A static variable, _ACTIONS_, which is a list of strings denoting each action.

* Implement a reward and transition function and pass them to MDP constructor (along with _ACTIONS_).

* I also suggest overwriting the "\_\_str\_\_" method of the class, and adding a "\_\_init\_\_.py" file to the directory.

* Create a State subclass for your MDP. I suggest overwriting the "\_\_hash\_\_", "\_\_eq\_\_", and "\_\_str\_\_" for the class to play along well with the agents.


## Making a New Agent

Make an Agent subclass, which requires:

* A method, _act(self, state, reward)_, that returns an action.

* A method, _reset()_, that puts the agent back to its _tabula rasa_ state.


Let me know if you have any questions or suggestions.

Cheers,

-Dave