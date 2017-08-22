''' FixedPolicyAgentClass.py: Class for a basic RL Agent '''

# Python imports.
from simple_rl.agents.AgentClass import Agent

class FixedPolicyAgent(Agent):
    ''' Agent Class with a fixed policy. '''

    NAME = "fixed-policy"

    def __init__(self, policy, name=NAME):
        '''
        Args:
            policy (func: S ---> A)
        '''
        Agent.__init__(self, name=name, actions=[])
        self.policy = policy

    def act(self, state, reward):
        '''
        Args:
            state (State): see StateClass.py
            reward (float): the reward associated with arriving in state @state.

        Returns:
            (str): action.
        '''
        return self.policy(state)

    def set_policy(self, new_policy):
        self.policy = new_policy

    def __str__(self):
        return str(self.name)
