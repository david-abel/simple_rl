# Grab classes.
from chain.ChainMDPClass import ChainMDP
from chain.ChainStateClass import ChainState
from grid_world.GridWorldMDPClass import GridWorldMDP
from grid_world.GridWorldStateClass import GridWorldState
from random.RandomMDPClass import RandomMDP
from random.RandomStateClass import RandomState
from taxi.TaxiOOMDPClass import TaxiOOMDP
from taxi.TaxiStateClass import TaxiState

# Only grab the ALE if it's installed.
import sys
if "ale_python_interface" in sys.modules:
	from atari.AtariMDPClass import AtariMDP
	from atari.AtariStateClass import AtariState
