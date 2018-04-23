from distutils.core import setup
from setuptools import find_packages
exec(open('simple_rl/_version.py').read())
setup(
  name = 'simple_rl',
  packages = find_packages(),
  scripts=['simple_rl/run_experiments.py'],
  version=__version__,
  description = 'A simple framework for experimenting with Reinforcement Learning in Python.',
  long_description = 'A simple framework for experimenting with Reinforcement Learning in Python.',
  author = 'David Abel',
  author_email = 'david_abel@brown.edu',
  url = 'https://github.com/david-abel/simple_rl',
  download_url = 'https://github.com/david-abel/simple_rl/tarball/' + str(__version__),
  keywords = ['Markov Decision Process', 'MDP', 'Reinforcement Learning'],
  classifiers = [],
)
