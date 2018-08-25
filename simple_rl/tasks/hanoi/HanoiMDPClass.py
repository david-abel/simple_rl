''' HanoiMDPClass.py: Contains a class for the classical planning/puzzle game Towers of Hanoi. '''

# Python imports
import itertools

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State

class HanoiMDP(MDP):
    ''' Class for a Tower of Hanoi MDP '''

    ACTIONS = ["01", "02", "10", "12", "20", "21"]

    def __init__(self, num_pegs=3, num_discs=3, gamma=0.95):
        '''
        Args:
            num_pegs (int)
            num_discs (int)
            gamma (float)
        '''
        self.num_pegs = num_pegs
        self.num_discs = num_discs
        HanoiMDP.ACTIONS = [str(x) + str(y) for x, y in itertools.product(range(self.num_pegs), range(self.num_pegs)) if x != y]

        # Setup init state.
        init_state = [" " for peg in range(num_pegs)]
        x = ""
        for i in range(num_discs):
            x += chr(97 + i)
        init_state[0] = x
        init_state = State(data=init_state)

        MDP.__init__(self, HanoiMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["num_pegs"] = self.num_pegs
        param_dict["num_discs"] = self.num_discs
   
        return param_dict

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        source_index = int(action[0])
        dest_index = int(action[1])

        return int(self._transition_func(state, action).is_terminal())

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''

        # Grab top discs on source and dest pegs.
        source_index = int(action[0])
        dest_index = int(action[1])
        source_top = state[source_index][-1]
        dest_top = state[dest_index][-1]

        # Make new state.        
        new_state_ls = state.get_data()[:]
        if dest_top < source_top:
            new_state_ls[source_index] = new_state_ls[source_index][:-1]
            if new_state_ls[source_index] == "":
                new_state_ls[source_index] = " "
            new_state_ls[dest_index] += source_top
            new_state_ls[dest_index] = new_state_ls[dest_index].replace(" ", "")
        new_state = State(new_state_ls)

        # Set terminal.
        if self._is_goal_state(state): # new_state[1] == "abc" or new_state[2] == "abc":
            new_state.set_terminal(True)

        return new_state

    def _is_goal_state(self, state):
        '''
        Args:
            state (simple_rl.State)

        Returns:
            (bool)
        '''
        for peg in state[1:]:
            if len(peg) == self.num_discs and sorted(peg) == list(peg):
                return True
        return False


    def __str__(self):
        return "hanoi"