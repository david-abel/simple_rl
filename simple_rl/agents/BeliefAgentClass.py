# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.pomdp.BeliefStateClass import BeliefState

class BeliefAgent(Agent):
    def __init__(self, name, actions, gamma=0.99):
        '''
        Args:
            name (str)
            actions (list)
            gamma (float
        '''
        Agent.__init__(self, name, actions, gamma)

    def act(self, belief_state, reward):
        '''

        Args:
            belief_state (BeliefState)
            reward (float)

        Returns:
            action (str)
        '''
        pass

    def policy(self, belief_state):
        '''
        Args:
            belief_state (BeliefState)

        Returns:
            action (str)
        '''
        return self.act(belief_state, 0)
