# Python imports.
import os

# Other imports.
from simple_rl.agents import Agent, RMaxAgent
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction

class AbstractionWrapper(Agent):

    def __init__(self,
                    SubAgentClass,
                    actions,
                    agent_params={},
                    state_abstr=None,
                    action_abstr=None,
                    name_ext="abstr"):
        '''
        Args:
            SubAgentClass (simple_rl.AgentClass)
            actions (list of str)
            agent_params (dict): A dictionary with key=param_name, val=param_value,
                to be given to the constructor for the instance of @SubAgentClass.
            state_abstr (StateAbstraction)
            state_abstr (ActionAbstraction)
            name_ext (str)
        '''

        # Setup the abstracted agent.
        self.agent = SubAgentClass(actions=actions, **agent_params)
        self.action_abstr = ActionAbstraction(prim_actions=self.agent.actions) if action_abstr is None else action_abstr
        self.state_abstr = StateAbstraction({}) if state_abstr is None else state_abstr

        Agent.__init__(self, name=self.agent.name + "-" + name_ext, actions=self.action_abstr.get_actions())

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        abstr_state = self.state_abstr.phi(ground_state)
        ground_action = self.action_abstr.act(self.agent, abstr_state, ground_state, reward)

        return ground_action

    def reset(self):
        # Write data.
        self.agent.reset()
        self.action_abstr.reset()

    def end_of_episode(self):
        self.agent.end_of_episode()
        self.action_abstr.end_of_episode()
