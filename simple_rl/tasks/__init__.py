# Python imports.
from __future__ import print_function

# Grab classes.
from simple_rl.tasks.bandit.BanditMDPClass import BanditMDP
from simple_rl.tasks.cart_pole.CartPoleMDPClass import CartPoleMDP
from simple_rl.tasks.cart_pole.CartPoleStateClass import CartPoleState
from simple_rl.tasks.chain.ChainMDPClass import ChainMDP
from simple_rl.tasks.chain.ChainStateClass import ChainState
from simple_rl.tasks.combo_lock.ComboLockMDPClass import ComboLockMDP
from simple_rl.tasks.four_room.FourRoomMDPClass import FourRoomMDP
from simple_rl.tasks.gather.GatherMDPClass import GatherMDP
from simple_rl.tasks.gather.GatherStateClass import GatherState
from simple_rl.tasks.grid_game.GridGameMDPClass import GridGameMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.tasks.hanoi.HanoiMDPClass import HanoiMDP
from simple_rl.tasks.navigation.NavigationWorldMDP import NavigationWorldMDP
from simple_rl.tasks.prisoners.PrisonersDilemmaMDPClass import PrisonersDilemmaMDP
from simple_rl.tasks.puddle.PuddleMDPClass import PuddleMDP
from simple_rl.tasks.random.RandomMDPClass import RandomMDP
from simple_rl.tasks.random.RandomStateClass import RandomState
from simple_rl.tasks.taxi.TaxiOOMDPClass import TaxiOOMDP
from simple_rl.tasks.taxi.TaxiStateClass import TaxiState
from simple_rl.tasks.trench.TrenchOOMDPClass import TrenchOOMDP
from simple_rl.tasks.rock_paper_scissors.RockPaperScissorsMDPClass import RockPaperScissorsMDP
try:
	from simple_rl.tasks.gym.GymMDPClass import GymMDP
except ImportError:
	print("Warning: OpenAI gym not installed.")
	pass
