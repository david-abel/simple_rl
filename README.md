# mdps
Some simple infrastructure for experimenting with Markov Decision Processes in Python 2.7.

See _experiments/runExperiments.py_ to run some basic experiments.

No irregular dependencies, just requires [numpy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/). 

## Overview

* (_tasks_): Implementations for a few standard MDPs

* (_experiments_): code for running experiments. runExperiments.py is the real meat: running that will autogenerate plots for a given set of {agents, MDP, parameters (numEpisodes, numInstances, etc.)}.

* (_agents_): code for some basic agents (a random actor, _Q_-learner, etc.). More to come (hopefully).

Cheers,

-Dave