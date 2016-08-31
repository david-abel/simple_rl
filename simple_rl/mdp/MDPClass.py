''' MDPClass.py: Contains the MDP Class. '''

class MDP(object):
    ''' Abstract class for a Markov Decision Process. '''
    def __init__(self, actions, transition_func, reward_func, init_state, gamma=0.95):
        self.actions = actions
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.gamma = gamma
        self.init_state = init_state
        self.cur_state = init_state

    def get_init_state(self):
        return self.init_state

    def get_curr_state(self):
        return self.cur_state

    def get_actions(self):
        return self.actions

    def get_gamma(self):
        return self.gamma

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            (tuple: <float,State>): reward, State
        '''

        reward = self.reward_func(self.cur_state, action)
        next_state = self.transition_func(self.cur_state, action)
        self.cur_state = next_state

        return reward, next_state

    def reset(self):
        self.cur_state = self.init_state
