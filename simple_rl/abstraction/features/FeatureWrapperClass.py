
# simple_rl imports.
from simple_rl.agents import Agent
from simple_rl.mdp import State

class FeatureWrapper(Agent):

    def __init__(self,
                    SubAgentClass,
                    agent_params={},
                    feature_mapper=None,
                    name_ext="-feats"):
        '''
        Args:
            SubAgentClass (simple_rl.AgentClass)
            agent_params (dict): A dictionary with key=param_name, val=param_value,
                to be given to the constructor for the instance of @SubAgentClass.
            feature_mapper (FeatureMapper)
            name_ext (str)
        '''

        # Setup the abstracted agent.
        self.agent = SubAgentClass(**agent_params)
        self.feature_mapper = feature_mapper

        if self.feature_mapper is not None:
            name_ext = "-" + feature_mapper.NAME if feature_mapper.NAME is not None else name_ext
        
        Agent.__init__(self, name=self.agent.name + name_ext, actions=self.agent.actions)

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''

        if self.feature_mapper is not None:
            new_state_feats = self.feature_mapper.get_features(ground_state)
            new_state = State(data=new_state_feats)
        else:
            new_state = ground_state

        action = self.agent.act(new_state, reward)

        return action

    def reset(self):
        self.agent.reset()

    def end_of_episode(self):
        self.agent.end_of_episode()
