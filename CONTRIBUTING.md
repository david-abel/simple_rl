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

Best,
Dave Abel (david_abel@brown.edu)
