from distutils.core import setup
setup(
  name = 'simple_rl',
  packages = ['simple_rl'],
  scripts=['run_experiments.py']
  install_requires=[
      'numpy',
      'sklearn',
      'matplotlib'
  ],
  version = '0.55',
  description = 'A simple framework for experimenting with Reinforcement Learning in Python 2.7',
  author = 'David Abel',
  author_email = 'dabel@cs.brown.com',
  url = 'https://github.com/david-abel/simple_rl',
  download_url = 'https://github.com/david-abel/simple_rl/tarball/v0.55',
  keywords = ['Markov Decision Process', 'MDP', 'Reinforcement Learning'],
  classifiers = [],
)
