''' MDPClass.py: Contains the MDP Class. '''

# Python imports.
import copy

class MDP(object):
    ''' Abstract class for a Markov Decision Process. '''
    
    def __init__(self, actions, transition_func, reward_func, init_state, gamma=0.99, step_cost=0):
        self.actions = actions
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.gamma = gamma
        self.init_state = copy.deepcopy(init_state)
        self.cur_state = init_state
        self.step_cost = step_cost

    # ---------------
    # -- Accessors --
    # ---------------

    def get_init_state(self):
        return self.init_state

    def get_curr_state(self):
        return self.cur_state

    def get_actions(self):
        return self.actions

    def get_gamma(self):
        return self.gamma

    def get_reward_func(self):
        return self.reward_func

    def get_transition_func(self):
        return self.transition_func

    def get_num_state_feats(self):
        return self.init_state.get_num_feats()

    def get_slip_prob(self):
        pass

    # --------------
    # -- Mutators --
    # --------------

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def set_step_cost(self, new_step_cost):
        self.step_cost = new_step_cost

    def set_slip_prob(self, slip_prob):
        pass

    # ----------
    # -- Core --
    # ----------

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            (tuple: <float,State>): reward, State

        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        '''
        reward = self.reward_func(self.cur_state, action)
        next_state = self.transition_func(self.cur_state, action)
        self.cur_state = next_state

        return reward, next_state

    def reset(self):
        self.cur_state = copy.deepcopy(self.init_state)

    def end_of_instance(self):
        pass
