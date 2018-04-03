# Python imports.
from os import path
import sys

# Other imports.
parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_dir)
import make_abstr_mdp
from state_abs import sa_helpers
from action_abs import aa_helpers
from action_abs.ActionAbstractionClass import ActionAbstraction
from simple_rl.utils import make_mdp
from HierarchyStateClass import HierarchyState
from StateAbstractionStackClass import StateAbstractionStack
from ActionAbstractionStackClass import ActionAbstractionStack

def make_hierarchy(mdp_distr, num_levels):
    '''
    Args:
        mdp_distr (MDPDistribution)
        num_levels (int)

    Returns:
        (tuple)
            1. StateAbstractionStack
            2. ActionAbstractionStack

    Notes:
        A one layer hierarchy is *flat* (that is, just uses the ground MDP). A
        two layer hierarchy has a single abstract level and the ground level.
    '''

    if num_levels <= 0:
        raise ValueError("(hiearchy_helpers.py) Error: @num_levels must be > 0 (given value: " + str(num_levels) + ").")

    sa_stack = StateAbstractionStack(list_of_phi=[])
    aa_stack = ActionAbstractionStack(list_of_aa=[], prim_actions=mdp_distr.get_actions())
    epsilon = 0.0

    for i in range(1, num_levels):
        print("\n" + "=" * 20)
        print("== Making layer " + str(i + 1) + " ==")
        print("=" * 20 + "\n")
        sa_stack, aa_stack, epsilon = add_layer(mdp_distr, sa_stack, aa_stack, init_epsilon=epsilon)
        # Update MDP Distribution

        epsilon += 0.90

    return sa_stack, aa_stack

def add_layer(mdp_distr, sa_stack, aa_stack, init_epsilon=0.0):
    '''
    Args:
        mdp_distr (MDPDistribution)
        sa_stack (StateAbstractionStack)
        aa_stack (ActionAbstractionStack)
        init_epsilon (float)

    Returns:
        (tuple):
            1. StateAbstractionStack
            2. ActionAbstractionStack
            3. (float): Final epsilon value.
    '''

    # Get next abstractions by iterating over compression ratio.
    epsilon, epsilon_incr = init_epsilon, 0.01

    while epsilon < 1.0:
        print("Abstraction rate (epsilon):", epsilon)

        # Set level to the largest shared between sa_stack and aa_stack.
        abstr_mdp_level = min(sa_stack.get_num_levels(), aa_stack.get_num_levels())
        sa_stack.set_level(abstr_mdp_level)
        aa_stack.set_level(abstr_mdp_level)

        # Add layer to state abstraction stack.
        sa_stack = add_layer_to_sa_stack(mdp_distr, sa_stack, aa_stack, epsilon)

        # Add layer to action abstraction stack.
        aa_stack, is_too_many_options = add_layer_to_aa_stack(mdp_distr, sa_stack, aa_stack)

        if is_too_many_options:
            # Too many options. Decrement and continue.
            epsilon += epsilon_incr
            sa_stack.remove_last_phi()
            continue
        else:
            break

    return sa_stack, aa_stack, epsilon

def add_layer_to_sa_stack(mdp_distr, sa_stack, aa_stack, epsilon):
    '''
    Args:
        mdp_distr (MDPDistribution)
        sa_stack (StateAbstractionStack)
        aa_stack (ActionAbstractionStack)
        epsilon (float)

    Returns:
        (StateAbstractionStack)
    '''

    # Check stack height.
    if sa_stack.get_num_levels() > 0:
        abstr_mdp_distr = make_abstr_mdp.make_abstr_mdp_distr_multi_level(mdp_distr, sa_stack, aa_stack)
    else:
        abstr_mdp_distr = mdp_distr

    # Make new phi.
    new_sa = sa_helpers.make_multitask_sa(abstr_mdp_distr, epsilon=epsilon)
    new_phi = _convert_abstr_states(new_sa._phi, sa_stack.get_num_levels() + 1)
    sa_stack.add_phi(new_phi)

    return sa_stack

def _convert_abstr_states(phi_dict, level):
    '''
    Args:
        phi_dict (dict)
        level (int)

    Returns:
        (phi_dict)

    Summary:
        Translates int based abstract states to HierarchyStates (which track their own level).
    '''
    for key in phi_dict.keys():
        state_num = phi_dict[key]
        phi_dict[key] = HierarchyState(data=state_num, level=level)

    return phi_dict

def add_layer_to_aa_stack(mdp_distr, sa_stack, aa_stack):
    '''
    Args:
        mdp_distr (MDPDistribution)
        sa_stack (StateAbstractionStack)
        aa_stack (ActionAbstractionStack)

    Returns:
        (tuple):
            1. (ActionAbstractionStack)
            2. (MDPDistribution)
            3. (bool)
    '''
    if aa_stack.get_num_levels() > 0:
        abstr_mdp_distr = make_abstr_mdp.make_abstr_mdp_distr_multi_level(mdp_distr, sa_stack, aa_stack)
    else:
        abstr_mdp_distr = mdp_distr

    # Make options for the level + 1 height.
    sa_stack.set_level_to_max()
    next_options = aa_helpers.get_directed_options_for_sa(abstr_mdp_distr, sa_stack, incl_self_loops=False, max_options=1024 / (aa_stack.get_num_levels() + 1))

    if not next_options:
        # Too many options, decrease abstracton ratio and continue.
        return aa_stack, True

    next_aa = ActionAbstraction(options=next_options, prim_actions=mdp_distr.get_actions())

    aa_stack.add_aa(next_aa)
    return aa_stack, False

def main():

    # ======================
    # == Make Environment ==
    # ======================
    mdp_class = "four_room"
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=7)
    actions = environment.get_actions()

    # ====================
    # == Make Hierarchy ==
    # ====================
    sa_stack, aa_stack = make_hierarchy(environment, num_levels=3)


if __name__ == "__main__":
    main()
