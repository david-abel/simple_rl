# Python imports.
from __future__ import print_function


# Grab classes.
from bandit.BanditMDPClass import BanditMDP
from chain.ChainMDPClass import ChainMDP
from chain.ChainStateClass import ChainState
from combo_lock.ComboLockMDPClass import ComboLockMDP
from hanoi.HanoiMDPClass import HanoiMDP
from grid_world.GridWorldMDPClass import GridWorldMDP
from grid_world.GridWorldStateClass import GridWorldState
from four_room.FourRoomMDPClass import FourRoomMDP
from random.RandomMDPClass import RandomMDP
from random.RandomStateClass import RandomState
from taxi.TaxiOOMDPClass import TaxiOOMDP
from taxi.TaxiStateClass import TaxiState
from prisoners.PrisonersDilemmaMDPClass import PrisonersDilemmaMDP
from rock_paper_scissors.RockPaperScissorsMDPClass import RockPaperScissorsMDP
from grid_game.GridGameMDPClass import GridGameMDP
try:
	from gym.GymMDPClass import GymMDP
except ImportError:
	print("Warning: OpenAI gym not installed.")
	pass