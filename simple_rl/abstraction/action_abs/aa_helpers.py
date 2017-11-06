# Other imports.
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.tasks import GridWorldMDP
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.InListPredicateClass import InListPredicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PolicyFromDictClass import PolicyFromDict

# ------------------------
# -- Goal Based Options --
# ------------------------
def make_goal_based_options(mdp_distr):
    '''
    Args:
        mdp_distr (MDPDistribution)

    Returns:
        (list): Contains Option instances.
    '''

    goal_list = set([])
    for mdp in mdp_distr.get_all_mdps():
        vi = ValueIteration(mdp)
        state_space = vi.get_states()
        for s in state_space:
            if s.is_terminal():
                goal_list.add(s)

    options = set([])
    for mdp in mdp_distr.get_all_mdps():

        init_predicate = Predicate(func=lambda x: True)
        term_predicate = InListPredicate(ls=goal_list)
        o = Option(init_predicate=init_predicate,
                    term_predicate=term_predicate,
                    policy=_make_mini_mdp_option_policy(mdp),
                    term_prob=0.0)
        options.add(o)

    return options

def _make_mini_mdp_option_policy(mini_mdp):
    '''
    Args:
        mini_mdp (MDP)

    Returns:
        Policy
    '''
    # Solve the MDP defined by the terminal abstract state.
    mini_mdp_vi = ValueIteration(mini_mdp, delta=0.001, max_iterations=1000, sample_rate=10)
    iters, val = mini_mdp_vi.run_vi()

    o_policy_dict = make_dict_from_lambda(mini_mdp_vi.policy, mini_mdp_vi.get_states())
    o_policy = PolicyFromDict(o_policy_dict)

    return o_policy.get_action

def make_dict_from_lambda(policy_func, state_list):
    policy_dict = {}
    for s in state_list:
        policy_dict[s] = policy_func(s)

    return policy_dict
