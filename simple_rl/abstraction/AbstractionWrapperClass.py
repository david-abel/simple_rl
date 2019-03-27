# Python imports.
import os

# Other imports.
from simple_rl.agents import Agent, RMaxAgent, FixedPolicyAgent
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction

class AbstractionWrapper(Agent):

    def __init__(self,
                    SubAgentClass,
                    agent_params={},
                    state_abstr=None,
                    action_abstr=None,
                    name_ext="-abstr"):
        '''
        Args:
            SubAgentClass (simple_rl.AgentClass)
            agent_params (dict): A dictionary with key=param_name, val=param_value,
                to be given to the constructor for the instance of @SubAgentClass.
            state_abstr (StateAbstraction)
            state_abstr (ActionAbstraction)
            name_ext (str)
        '''

        # Setup the abstracted agent.
        self.agent = SubAgentClass(**agent_params)
        self.action_abstr = action_abstr
        self.state_abstr = state_abstr
        all_actions = self.action_abstr.get_actions() if self.action_abstr is not None else self.agent.actions
        
        Agent.__init__(self, name=self.agent.name + name_ext, actions=all_actions)

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''

        if self.state_abstr is not None:
            abstr_state = self.state_abstr.phi(ground_state)
        else:
            abstr_state = ground_state

        if self.action_abstr is not None:
            ground_action = self.action_abstr.act(self.agent, abstr_state, ground_state, reward)
        else:
            ground_action = self.agent.act(abstr_state, reward)

        return ground_action

    def reset(self):
        # Write data.
        self.agent.reset()

        if self.action_abstr is not None:
            self.action_abstr.reset()

    def end_of_episode(self):
        self.agent.end_of_episode()
        if self.action_abstr is not None:
            self.action_abstr.end_of_episode()
