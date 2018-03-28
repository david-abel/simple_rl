# Python imports.
import Queue

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.planning.ValueIterationClass import ValueIteration
import hierarchy_helpers

class HierarchicalValueIteration(ValueIteration):

    def __init__(self, mdp, rew_func_list, trans_func_list, sa_stack, aa_stack, name="hierarch_value_iter", delta=0.001, max_iterations=200, sample_rate=3):
        '''
        Args:
            mdp (MDP)
            delta (float): After an iteration if VI, if no change more than @\delta has occurred, terminates.
            max_iterations (int): Hard limit for number of iterations.
            sample_rate (int): Determines how many samples from @mdp to take to estimate T(s' | s, a).
            horizon (int): Number of steps before terminating.
        '''
        self.rew_func_list = rew_func_list
        self.trans_func_list = trans_func_list
        self.sa_stack = sa_stack
        self.aa_stack = aa_stack
        abstr_actions = []

        for aa in self.aa_stack.get_aa_list():
            abstr_actions += aa.get_actions()
        self.actions = mdp.get_actions() + abstr_actions   

        ValueIteration.__init__(self, mdp, name=name, delta=delta, max_iterations=max_iterations, sample_rate=sample_rate)

    def _compute_matrix_from_trans_func(self):
        if self.has_computed_matrix:
            # We've already run this, just return.
            return

        self.trans_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
            # K: state
                # K: a
                    # K: s_prime
                    # V: prob

        for s in self.get_states():
            for a in self.actions:
                for sample in range(self.sample_rate):
                    s_prime = self.transition_func(s, a)
                    self.trans_dict[s][a][s_prime] += 1.0 / self.sample_rate

        self.has_computed_matrix = True

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''
        state_queue = Queue.Queue()

        for lvl in range(self.sa_stack.get_num_levels()):
            abstr_state = self.sa_stack.phi(self.init_state, lvl)
            self.states.add(abstr_state)
            state_queue.put(abstr_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in range(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    next_state = self.transition_func(s,a) # Need to use T w.r.t. the abstract MDP...

                    if next_state not in self.states:
                        self.states.add(next_state)
                        state_queue.put(next_state)

        self.reachability_done = True

def main():


    # ========================
    # === Make Environment ===
    # ========================
    mdp_class = "four_room"
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=10)
    actions = environment.get_actions()

    # ==========================
    # === Make SA, AA Stacks ===
    # ==========================
    # sa_stack, aa_stack = aa_stack_h.make_random_sa_diropt_aa_stack(environment, max_num_levels=3)
    sa_stack, aa_stack = hierarchy_helpers.make_hierarchy(environment, num_levels=3)

    mdp = environment.sample()
    HVI = HierarchicalValueIteration(mdp, sa_stack, aa_stack)
    VI = ValueIteration(mdp)

    h_iters, h_val = HVI.run_vi()
    iters, val = VI.run_vi()

if __name__ == "__main__":
    main()