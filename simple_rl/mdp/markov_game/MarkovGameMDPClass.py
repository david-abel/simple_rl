''' MarkovGameMDP.py: Contains implementation for simple Markov Games. '''

from ...mdp.MDPClass import MDP

class MarkovGameMDP(MDP):
    ''' Abstract class for a Markov Decision Process. '''
    def __init__(self, actions, transition_func, reward_func, init_state, gamma=0.95, num_agents=2):
        MDP.__init__(self, actions, transition_func, reward_func, init_state=init_state, gamma=gamma)
        self.num_agents = num_agents

    def execute_agent_action(self, action_dict):
        '''
        Args:
            actions (dict): an action for each agent.
        '''
        if len(action_dict.keys()) != self.num_agents:
            print "Error: only", len(action_dict.keys()), "action(s) was/were provided, but there are", self.num_agents, "agents."
            quit()

        reward_dict = self.reward_func(self.cur_state, action_dict)
        next_state = self.transition_func(self.cur_state, action_dict)
        self.cur_state = next_state

        return reward_dict, next_state