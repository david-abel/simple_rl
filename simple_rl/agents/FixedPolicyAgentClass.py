''' FixedPolicyAgentClass.py: Class for a basic RL Agent '''

# Python imports.
from AgentClass import Agent

class FixedPolicyAgent(Agent):
    ''' Agent Class with a fixed policy. '''

    def __init__(self, policy, name="fixed-policy"):
        '''
        Args:
            policy (func: S ---> A)
        '''
        Agent.__init__(self, name=name, actions=[])
        self.policy = policy
        self.name = name

    def act(self, state, reward):
        '''
        Args:
            state (State): see StateClass.py
            reward (float): the reward associated with arriving in state @state.

        Returns:
            (str): action.
        '''
        return self.policy(state)

    def __str__(self):
        return str(self.name)
