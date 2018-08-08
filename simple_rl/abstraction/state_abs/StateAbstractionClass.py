# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP

class StateAbstraction(object):

    def __init__(self, phi=None, ground_state_space=[]):
        '''
        Args:
            phi (dict)
        '''
        # key:state, val:int. (int represents an abstract state).
        self._phi = phi if phi is not None else {s_g: s_g for s_g in ground_state_space}

    def set_phi(self, new_phi):
        self._phi = new_phi

    def phi(self, state):
        '''
        Args:
            state (State)

        Returns:
            state (State)
        '''

        # Check types.
        if state not in self._phi.keys():
            raise KeyError

        if not isinstance(self._phi[state], State):
            raise TypeError

        # Get abstract state.
        abstr_state = self._phi[state]
        abstr_state.set_terminal(state.is_terminal())

        return abstr_state

    def make_cluster(self, list_of_ground_states):
        if len(list_of_ground_states) == 0:
            return

        abstract_value = 0
        if len(self._phi.values()) != 0:
            abstract_value = max(self._phi.values()) + 1

        for state in list_of_ground_states:
            self._phi[state] = abstract_value

    def get_ground_states_in_abs_state(self, abs_state):
        '''
        Args:
            abs_state (State)

        Returns:
            (list): Contains all ground states in the cluster.
        '''
        return [s_g for s_g in self.get_ground_states() if self.phi(s_g) == abs_state]
    
    def get_lower_states_in_abs_state(self, abs_state):
        '''
        Args:
            abs_state (State)

        Returns:
            (list): Contains all ground states in the cluster.

        Notes:
            Here to simplify the state abstraction stack subclass.
        '''
        return self.get_ground_states_in_abs_state(abs_state)

    def get_abs_states(self):
        # For each ground state, get its abstract state.
        return set([self.phi(val) for val in set(self._phi.keys())])

    def get_abs_cluster_num(self, abs_state):
        return list(set(self._phi.values())).index(abs_state.data)

    def get_ground_states(self):
        return self._phi.keys()

    def get_lower_states(self):
        return self.get_ground_states()

    def get_num_abstr_states(self):
        return len(set(self._phi.values()))

    def get_num_ground_states(self):
        return len(set(self._phi.keys()))

    def reset(self):
        self._phi = {}

    def __add__(self, other_abs):
        '''
        Args:
            other_abs
        '''
        merged_state_abs = {}

        # Move the phi into a cluster dictionary.
        cluster_dict = defaultdict(list)
        for k, v in self._phi.iteritems():
            # Cluster dict: v is abstract, key is ground.
            cluster_dict[v].append(k)

        # Move the phi into a cluster dictionary.
        other_cluster_dict = defaultdict(list)
        for k, v in other_abs._phi.iteritems():
            other_cluster_dict[v].append(k)


        for ground_state in self._phi.keys():
            

            # Get the two clusters associated with a state.
            states_cluster = self._phi[ground_state]
            if ground_state in other_abs._phi.keys():
                # Only add if it's in both clusters.
                states_other_cluster = other_abs._phi[ground_state]
            else:
                continue

            for s_g in cluster_dict[states_cluster]:
                if s_g in other_cluster_dict[states_other_cluster]:
                    # Every ground state that's in both clusters, merge.
                    merged_state_abs[s_g] = states_cluster

        new_sa = StateAbstraction(phi=merged_state_abs)

        return new_sa
