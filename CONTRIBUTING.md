# Contributing to simple_rl

Thanks for the interest! I've here put together a quick guide for contributing to the library.

As of August 2018, the standard pipeline for contributing is as follows:
  * Please follow package and coding conventions (see below).
  * If you add something substantive, please do the following:
    * Run the [basic testing script](https://github.com/david-abel/simple_rl/blob/master/tests/basic_test.py) and ensure all tests pass in both Python 2 and Python 3.
    * If you decide your contribution would benefit from a new example/test, please write a quick example file to put in the [examples directory](https://github.com/david-abel/simple_rl/tree/master/examples).
  * Issue a pull request to the main branch, which I will review as quickly as I can. If I haven't approved the request in three days, feel free to email me.

# Library Standards

Please ensure:
  * Your code is compatible with both Python 2 and Python 3.
  * If you add any deep learning, the library will be moving toward [PyTorch](https://pytorch.org/) as its standard.
  * I encourage the use of [https://www.pylint.org/](pylint).
  * Please include a brief log message for all commits (ex: use -m "message").
  * Files are all named lower case with underscores between words *unless* that file contains a Class.
  * Class files are named with PascalCase (so all words are capitalized) with the last word being "Class" (ex: QLearningAgentClass.py).

## Coding conventions

Please:
  * Indent with spaces.
  * Spaces after all list items and algebraic operators (ex: ["a", "b", 5 + 6]).
  * Doc-strings follow the (now deprecated, sadly) [Google doc-string format](https://google.github.io/styleguide/pyguide.html#Comments). Please use this until this contribution guide says otherwise.
  * Separate standard python imports from non-python imports at the top of each file, with python imports appearing first.

## Things to Work On

If you'd like to help add on to the library, here are the key ways I'm hoping to extend its current features:
* __Planning__: Finish MCTS [[Coloum 2006]](https://hal.inria.fr/file/index/docid/116992/filename/CG2006.pdf), implement RTDP [[Barto et al. 1995]](https://pdfs.semanticscholar.org/2838/e01572bf53805c502ec31e3e00a8e1e0afcf.pdf)
* __Deep RL__: Write a DQN [[Mnih et al. 2015]](http://www.davidqiu.com:8888/research/nature14236.pdf) in PyTorch, possibly others (some kind of policy gradient).
* __Efficiency__: Convert most defaultdict/dict uses to numpy.
* __Reproducibility__: The new reproduce feature is limited in scope -- I'd love for someone to extend it to work with OO-MDPs, Planning, MarkovGames, POMDPs, and beyond.
* __Docs__: Write a nice tutorial and give thorough documentation.
* __Visuals__: Unify MDP visualization.
* __Misc__: Additional testing.


Best,
Dave Abel (david_abel@brown.edu)
