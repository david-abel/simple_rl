# Python imports.
import random
from os import path
import sys

# Other imports.
from HierarchyStateClass import HierarchyState
from simple_rl.utils import make_mdp
from simple_rl.planning.ValueIterationClass import ValueIteration
parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_dir)
from state_abs.StateAbstractionClass import StateAbstraction
from state_abs import sa_helpers
from StateAbstractionStackClass import StateAbstractionStack
import make_abstr_mdp



# ----------------------------------
# -- Make State Abstraction Stack --
# ----------------------------------

def make_random_sa_stack(mdp_distr, cluster_size_ratio=0.5, max_num_levels=2):
    '''
    Args:
        mdp_distr (MDPDistribution)
        cluster_size_ratio (float): A float in (0,1) that determines the size of the abstract state space.
        max_num_levels (int): Determines the _total_ number of levels in the hierarchy (includes ground).

    Returns:
        (StateAbstraction)
    '''

    # Get ground state space.
    vi = ValueIteration(mdp_distr.get_all_mdps()[0], delta=0.0001, max_iterations=5000)
    ground_state_space = vi.get_states()
    sa_stack = StateAbstractionStack(list_of_phi=[])

    # Each loop adds a stack.
    for i in range(max_num_levels - 1):

        # Grab curent state space (at level i).
        cur_state_space = _get_level_i_state_space(ground_state_space, sa_stack, i)
        cur_state_space_size = len(cur_state_space)

        if int(cur_state_space_size / cluster_size_ratio) <= 1:
            # The abstract is as small as it can get.
            break

        # Add the mapping.
        new_phi = {}
        for s in cur_state_space:
            new_phi[s] = HierarchyState(data=random.randint(1, max(int(cur_state_space_size * cluster_size_ratio), 1)), level=i + 1)

        if len(set(new_phi.values())) <= 1:
            # The abstract is as small as it can get.
            break

        # Add the sa to the stack.
        sa_stack.add_phi(new_phi)

    return sa_stack

def _get_level_i_state_space(ground_state_space, state_abstr_stack, level):
    '''
    Args:
        mdp_distr (MDPDistribution)
        state_abstr_stack (StateAbstractionStack)
        level (int)

    Returns:
        (list)
    '''
    level_i_state_space = set([])
    for s in ground_state_space:
        level_i_state_space.add(state_abstr_stack.phi(s, level))

    return list(level_i_state_space)


def main():
    # Make MDP Distribution.
    mdp_class = "four_room"
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=10)

    sa_stack = make_random_sa_stack(environment, max_num_levels=5)
    sa_stack.print_state_space_sizes()


if __name__ == "__main__":
    main()