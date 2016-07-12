# mdps
Some simple infrastructure for experimenting with Markov Decision Processes in Python 2.7.

See _experiments/runExperiments.py_ to run some basic experiments.

Just requires [numpy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/). 

## Overview

* (_tasks_): Implementations for a few standard MDPs

* (_experiments_): code for running experiments. runExperiments.py is the real meat: running that will autogenerate plots for a given set of {agents, MDP, parameters (numEpisodes, numInstances, etc.)}.

* (_agents_): code for some basic agents (a random actor, _Q_-learner, etc.). More to come (hopefully).

## Making a New MDP

Make a directory in _tasks_. Then make an subclass MDP:

* Add a static variabel consisting of a list of strings _actions_, denoting each action.

* Implement a reward and transition function and pass them to MDP constructor (along with _actions_).

* I also suggest overwriting the "__str__" method of the class, and adding a "__init__.py" file to the directory.

* Create a State subclass for your MDP. I suggest overwriting the "__hash__", "__eq__", and "__str__".


## Making a New Agent

Make an Agent subclass in _agents/_. Just needs:

* A method, _act(self, state, reward)_, that returns an action.

* A method, _reset()_, that puts the agent back to its _tabula rasa_ state.

Let me know if you have any questions or suggestions.

Cheers,

-Dave