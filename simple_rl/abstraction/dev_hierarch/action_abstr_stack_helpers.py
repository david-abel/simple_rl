# Python imports.
from __future__ import print_function
from os import path
import sys

# Other imports.
parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_dir)
import state_abstr_stack_helpers as sa_stack_helpers
from StateAbstractionStackClass import StateAbstractionStack
from ActionAbstractionStackClass import ActionAbstractionStack
from action_abs.ActionAbstractionClass import ActionAbstraction
from action_abs import aa_helpers
from simple_rl.utils import make_mdp


def make_random_sa_diropt_aa_stack(mdp_distr, max_num_levels=3):
    '''
    Args:
        mdp_distr (MDPDistribution)
        max_num_levels (int)

    Returns:
        (tuple):
            1. StateAbstraction
            2. ActionAbstraction
    '''

    # Get Abstractions by iterating over compression ratio.
    cluster_size_ratio, ratio_decr = 0.3, 0.05

    while cluster_size_ratio > 0.001:
        print("Abstraction ratio:", cluster_size_ratio)

        # Make State Abstraction stack.
        sa_stack = sa_stack_helpers.make_random_sa_stack(mdp_distr, cluster_size_ratio=cluster_size_ratio, max_num_levels=max_num_levels)
        sa_stack.print_state_space_sizes()

        # Make action abstraction stack.
        aa_stack = make_directed_options_aa_from_sa_stack(mdp_distr, sa_stack)

        if not aa_stack:
            # Too many options. Decrement and continue.
            cluster_size_ratio -= ratio_decr
            continue
        else:
            break

    return sa_stack, aa_stack

# ----------------------
# -- Directed Options --
# ----------------------

def make_directed_options_aa_from_sa_stack(mdp_distr, sa_stack):
    '''
    Args:
        mdp_distr (MDPDistribution)
        sa_stack (StateAbstractionStack)

    Returns:
        (ActionAbstraction)
    '''

    aa_stack = ActionAbstractionStack(list_of_aa=[], prim_actions=mdp_distr.get_actions())

    for level in range(1, sa_stack.get_num_levels() + 1):
        
        # Make directed options for the current level.
        sa_stack.set_level(level)
        next_options = aa_helpers.get_directed_options_for_sa(mdp_distr, sa_stack, incl_self_loops=False)

        if not next_options:
            # Too many options, decrease abstracton ratio and continue.
            return False

        next_aa = ActionAbstraction(options=next_options, prim_actions=mdp_distr.get_actions())
        aa_stack.add_aa(next_aa)

    return aa_stack


def main():
    # Make MDP Distribution.
    mdp_class = "four_room"
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=10)

    make_random_sa_diropt_aa_stack(environment, max_num_levels=3)

if __name__ == "__main__":
    main()
