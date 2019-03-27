# simple_rl
A simple framework for experimenting with Reinforcement Learning in Python.

There are loads of other great libraries out there for RL. The aim of this one is twofold:

1. Simplicity.
2. Reproducibility of results.

A brief tutorial for a slightly earlier version is available [here](http://cs.brown.edu/~dabel/blog/posts/simple_rl.html). As of version 0.77, the library should work with both Python 2 and Python 3. Please let me know if you find that is not the case!

simple_rl requires [numpy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/). Some MDPs have visuals, too, which requires [pygame](http://www.pygame.org/news). Also includes support for hooking into any of the [Open AI Gym environments](https://gym.openai.com/envs). The library comes along with basic test script, contained in the _tests_ directory. I suggest running it and making sure all tests pass when you install the library.

[Documentation available here](https://david-abel.github.io/simple_rl/docs/index.html)

## Installation

The easiest way to install is with [pip](https://pypi.python.org/pypi/pip). Just run:

	pip install simple_rl

Alternatively, you can download simple_rl [here](https://github.com/david-abel/simple_rl/tarball/v0.811).

## New Feature: Easy Reproduction of Results

I just added a new feature I'm quite excited about: *easy reproduction of results*. Every experiment run now outputs a file "full_experiment.txt" in the _results/exp_name/_ directory. The new function _reproduce_from_exp_file(file_name)_, when pointed at an experiment directory, will reassemble and rerun an entire experiment based on this file. The goal here is to encourage simple tracking of experiments and enable quick result-reproduction. It only works with MDPs though -- it does not yet work with OOMDPs, POMDPs, or MarkovGames (I'd be delighted if someone wants to make it work, though!).

See the second example below for a quick sense of how to use this feature.

## Example

Some examples showcasing basic functionality are included in the [examples](https://github.com/david-abel/simple_rl/tree/master/examples) directory.

To run a simple experiment, import the _run_agents_on_mdp(agent_list, mdp)_ method from _simple_rl.run_experiments_ and call it with some agents for a given MDP. For example:

	# Imports
	from simple_rl.run_experiments import run_agents_on_mdp
	from simple_rl.tasks import GridWorldMDP
	from simple_rl.agents import QLearningAgent

	# Run Experiment
	mdp = GridWorldMDP()
	agent = QLearningAgent(mdp.get_actions())
	run_agents_on_mdp([agent], mdp)

Running the above code will run _Q_-learning on a simple GridWorld. When it finishes it stores the results in _cur_dir/results/*_ and makes and opens the following plot:

<img src="https://david-abel.github.io/blog/posts/images/simple_grid.jpg" width="480" align="center">

For a slightly more complicated example, take a look at the code of _simple_example.py_. Here we run two agents on the grid world from the Russell-Norvig AI textbook:

	from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
	from simple_rl.tasks import GridWorldMDP
	from simple_rl.run_experiments import run_agents_on_mdp

    # Setup MDP.
    mdp = GridWorldMDP(width=4, height=3, init_loc=(1, 1), goal_locs=[(4, 3)], lava_locs=[(4, 2)], gamma=0.95, walls=[(2, 2)], slip_prob=0.05)

    # Setup Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rmax_agent = RMaxAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rmax_agent, rand_agent], mdp, instances=5, episodes=50, steps=10)

The above code will generate the following plot:

<img src="https://david-abel.github.io/blog/posts/images/rn_grid.jpg" width="480" align="center">

To showcase the new reproducibility feature, suppose we now wanted to reproduce the above experiment. We just do the following:

	from simple_rl.run_experiments import reproduce_from_exp_file

	reproduce_from_exp_file("gridworld_h-3_w-4")

Which will rerun the entire experiment, based on a file created and populated behind the scenes. Then, we should get the following plot:

<img src="https://david-abel.github.io/blog/posts/images/rn_grid_reproduce.jpg" width="480" align="center">

Easy! This is a new feature, so there may be bugs -- just let me know as things come up. It's only supposed to work for MDPs, not POMDPs/OOMDPs/MarkovGameMDPs (so far). Take a look at [_reproduce_example.py_](https://github.com/david-abel/simple_rl/blob/master/examples/reproduce_example.py) for a bit more detail.

## Overview

* (_agents_): Code for some basic agents (a random actor, _Q_-learning, [[R-Max]](http://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf), _Q_-learning with a Linear Approximator, and so on).

* (_experiments_): Code for an Experiment class to track parameters and reproduce results.

* (_mdp_): Code for a basic MDP and MDPState class, and an MDPDistribution class (for  lifelong learning). Also contains OO-MDP implementation [[Diuk et al. 2008]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.7056&rep=rep1&type=pdf).

* (_planning_): Implementations for planning algorithms, includes ValueIteration and MCTS [[Couloum 2006]](https://hal.archives-ouvertes.fr/file/index/docid/116992/filename/CG2006.pdf), the latter being still in development.

* (_tasks_): Implementations for a few standard MDPs (grid world, N-chain, Taxi [[Dietterich 2000]](http://www.scs.cmu.edu/afs/cs/project/jair/pub/volume13/dietterich00a.pdf), and the [OpenAI Gym](https://gym.openai.com/envs)).

* (_utils_): Code for charting and other utilities.

## Contributing

If you'd like to contribute: that's great! Take a look at some of the needed improvements below: I'd love for folks to work on those items. Please see the [contribution guidelines](https://github.com/david-abel/simple_rl/blob/master/CONTRIBUTING.md). Email me with any questions.

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

I'm hoping to add the following features:

* __Planning__: Finish MCTS [[Coloum 2006]](https://hal.inria.fr/file/index/docid/116992/filename/CG2006.pdf), implement RTDP [[Barto et al. 1995]](https://pdfs.semanticscholar.org/2838/e01572bf53805c502ec31e3e00a8e1e0afcf.pdf)
* __Deep RL__: Write a DQN [[Mnih et al. 2015]](http://www.davidqiu.com:8888/research/nature14236.pdf) in PyTorch, possibly others (some kind of policy gradient).
* __Efficiency__: Convert most defaultdict/dict uses to numpy.
* __Reproducibility__: The new reproduce feature is limited in scope -- I'd love for someone to extend it to work with OO-MDPs, Planning, MarkovGames, POMDPs, and beyond.
* __Docs__: Tutorial and documentation.
* __Visuals__: Unify MDP visualization.
* __Misc__: Additional testing.


Cheers,

-Dave