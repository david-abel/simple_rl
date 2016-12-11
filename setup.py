from distutils.core import setup
setup(
  name = 'simple_rl',
  packages = ['simple_rl', 'simple_rl.utils', 'simple_rl.mdp', 'simple_rl.mdp.oomdp',\
  'simple_rl.agents', 'simple_rl.experiments', 'simple_rl.tasks',\
  'simple_rl.tasks.chain', 'simple_rl.tasks.random', 'simple_rl.tasks.atari',\
  'simple_rl.tasks.grid_world', 'simple_rl.tasks.taxi'],
  scripts=['simple_rl/run_experiments.py'],
  install_requires=[
      'numpy',
      'sklearn',
      'matplotlib'
  ],
  version = '0.6',
  description = 'A simple framework for experimenting with Reinforcement Learning in Python 2.7',
  author = 'David Abel',
  author_email = 'dabel@cs.brown.com',
  url = 'https://github.com/david-abel/simple_rl',
  download_url = 'https://github.com/david-abel/simple_rl/tarball/v0.6',
  keywords = ['Markov Decision Process', 'MDP', 'Reinforcement Learning'],
  classifiers = [],
)
