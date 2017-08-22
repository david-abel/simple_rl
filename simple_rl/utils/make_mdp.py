# Python imports.
import itertools
import random
from collections import defaultdict

# Other imports.
from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP, FourRoomMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.mdp import MDPDistribution

def make_markov_game(markov_game_class="grid_game"):
    return {"prison":PrisonersDilemmaMDP(),
            "rps":RockPaperScissorsMDP(),
            "grid_game":GridGameMDP()}[markov_game_class]

def make_mdp(mdp_class="grid", state_size=7):
    '''
    Returns:
        (MDP)
    '''
    # Grid/Hallway stuff.
    width, height = state_size, state_size
    hall_goal_locs = [(i, width) for i in range(1, height+1)]

    # Taxi stuff.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":state_size / 2, "y":state_size / 2, "dest_x":state_size-2, "dest_y":2, "in_taxi":0}]
    walls = []


    mdp = {"hall":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=hall_goal_locs),
            "pblocks_grid":make_grid_world_from_file("pblocks_grid.txt", randomize=True),
            "grid":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=[(state_size, state_size)]),
            "four_room":FourRoomMDP(width=width, height=height, goal_locs=[(width,height)]),
            "chain":ChainMDP(num_states=state_size),
            "random":RandomMDP(num_states=50, num_rand_trans=2),
            "taxi":TaxiOOMDP(width=state_size, height=state_size, slip_prob=0.0, agent=agent, walls=walls, passengers=passengers)}[mdp_class]

    return mdp

def make_mdp_distr(mdp_class="grid", grid_dim=7, horizon=0):
    '''
    Args:
        mdp_class (str): one of {"grid", "random"}
        horizon (int)

    Returns:
        (MDPDistribution)
    '''
    mdp_dist_dict = {}
    height, width = grid_dim, grid_dim

    # Define goal locations.
        
        # Corridor.
    corr_width = 20
    corr_goal_magnitude = random.randint(1, 5)
    corr_goal_cols = [i for i in xrange(1, corr_goal_magnitude)] + [j for j in xrange(corr_width-corr_goal_magnitude, corr_width + 1)]
    corr_goal_locs  = list(itertools.product(corr_goal_cols, [1]))

        # Grid World
    grid_world_rows, grid_world_cols = [i for i in xrange(width - 4, width)], [j for j in xrange(height - 4, height)]
    grid_goal_locs = list(itertools.product(grid_world_rows, grid_world_cols))

        # Hallway.
    hall_goal_locs = [(i, width) for i in range(1, height + 1)]

        # Four room.
    four_room_goal_locs = [(2,2), (width, height), (width, 1), (1, height)]

        # Taxi.
    agent = {"x":1, "y":1, "has_passenger":0}
    walls = []

    goal_loc_dict = {"four_room":four_room_goal_locs,
                    "hall":hall_goal_locs,
                    "grid":grid_goal_locs,
                    "corridor":corr_goal_locs
                    }
    
    # MDP Probability.
    num_mdps = 10 if mdp_class not in goal_loc_dict.keys() else len(goal_loc_dict[mdp_class])
    mdp_prob = 1.0 / num_mdps

    for i in range(num_mdps):

        new_mdp = {"hall":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=[goal_loc_dict["hall"][i % len(goal_loc_dict["hall"])]]),
                    "corridor":GridWorldMDP(width=20, height=1, init_loc=(10, 1), goal_locs=[goal_loc_dict["corridor"][i % len(goal_loc_dict["corridor"])]], is_goal_terminal=True),
                    "grid":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=[goal_loc_dict["grid"][i % len(goal_loc_dict["grid"])]], is_goal_terminal=True),
                    "four_room":FourRoomMDP(width=width, height=height, goal_locs=[goal_loc_dict["four_room"][i % len(goal_loc_dict["four_room"])]]),
                    # THESE GOALS ARE SPECIFIED IMPLICITLY:
                    "pblocks_grid":make_grid_world_from_file("pblocks_grid.txt", randomize=True),
                    "chain":ChainMDP(num_states=10, reset_val=random.choice([0, 0.01, 0.05, 0.1, 0.2, 0.5])),
                    "random":RandomMDP(num_states=40, num_rand_trans=random.randint(1,10)),
                    "taxi":TaxiOOMDP(4, 4, slip_prob=0.0, agent=agent, walls=walls, \
                                    passengers=[{"x":2, "y":2, "dest_x":random.randint(1,4), "dest_y":random.randint(1,4), "in_taxi":0}])}[mdp_class]

        mdp_dist_dict[new_mdp] = mdp_prob

    return MDPDistribution(mdp_dist_dict, horizon=horizon)
