'''
Basic LinUCB implementation.
'''

# Python imports.
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent

class LinUCBAgent(Agent):
    '''
    From:
        Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
        News Article Recommendation." In Proceedings of the 19th
        International Conference on World Wide Web (WWW), 2010.
    '''

    def __init__(self, actions, name="LinUCB", rand_init=True, context_size=1, alpha=1.5):
        '''
        Args:
            actions (list): Contains a string for each action.
            name (str)
            context_size (int)
            alpha (float): Uncertainty parameter.
        '''
        Agent.__init__(self, name, actions)
        self.alpha = alpha
        self.context_size = context_size
        self.prev_context = None
        self.step_number = 0
        self.rand_init = rand_init
        self._init_action_model(rand_init)


    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        
        param_dict["rand_init"] = self.rand_init
        param_dict["context_size"] = self.context_size
        param_dict["alpha"] = self.alpha

        return param_dict

    def _init_action_model(self, rand_init=True):
        '''
        Summary:
            Initializes model parameters
        '''
        self.model = {'act': {}, 'act_inv': {}, 'theta': {}, 'b': {}}
        for action_id in range(len(self.actions)):
            self.model['act'][action_id] = np.identity(self.context_size)
            self.model['act_inv'][action_id] = np.identity(self.context_size)
            if rand_init:
                self.model['theta'][action_id] = np.random.random((self.context_size, 1))
            else:
                self.model['theta'][action_id] = np.zeros((self.context_size, 1))
            self.model['b'][action_id] = np.zeros((self.context_size,1))

    def _compute_score(self, context):
        '''
        Args:
            context (list)

        Returns:
            (dict):
                K (str): action
                V (float): score
        '''

        a_inv = self.model['act_inv']
        theta = self.model['theta']

        estimated_reward = {}
        uncertainty = {}
        score_dict = {}
        max_score = 0
        for action_id in range(len(self.actions)):
            action_context = np.reshape(context[action_id], (-1, 1))
            estimated_reward[action_id] = float(theta[action_id].T.dot(action_context))
            uncertainty[action_id] = float(self.alpha * np.sqrt(action_context.T.dot(a_inv[action_id]).dot(action_context)))
            score_dict[action_id] = estimated_reward[action_id] + uncertainty[action_id]

        return score_dict

    def update(self, reward):
        '''
        Args:
            reward (float)

        Summary:
            Updates self.model according to self.prev_context, self.prev_action, @reward.
        '''
        action_id = self.actions.index(self.prev_action)
        action_context = np.reshape(self.prev_context[action_id], (-1, 1))
        self.model['act'][action_id] += action_context.dot(action_context.T)
        self.model['act_inv'][action_id] = np.linalg.inv(self.model['act'][action_id])
        self.model['b'][action_id] += reward * action_context
        self.model['theta'][action_id] = self.model['act_inv'][action_id].dot(self.model['b'][action_id])

    def act(self, context, reward):
        '''
        Args:
            context (iterable)
            reward (float)

        Returns:
            (str): action.
        '''

        # Update previous context-action pair.
        if self.prev_action is not None:
            self.update(reward)

        # Compute score.
        context = self._pre_process_context(context)
        score = self._compute_score(context)

        # Compute best action.
        best_action = np.random.choice(self.actions)
        max_score = float("-inf")
        for action_id in range(len(self.actions)):
            if score[action_id] > max_score:
                max_score = score[action_id]
                best_action = self.actions[action_id]


        # Update prev pointers.
        self.prev_action = best_action
        self.prev_context = context
        self.step_number += 1
        
        return best_action

    def _pre_process_context(self, context):
        if context.get_num_feats() == 1:
            # If there's no context (that is, we're just in a regular bandit).
            context = context.features()

        if not hasattr(context[0], '__iter__'):
            # If we only have a single context.
            new_context = {}
            for action_id in range(len(self.actions)):
                new_context[action_id] = context
            context = new_context

        return context
