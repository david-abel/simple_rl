'''
Basic LinUCB implementation.
'''

# Python imports.
import numpy as np

# Local imports.
from ..AgentClass import Agent

class LinUCBAgent(Agent):
    '''
    From:
        Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
        News Article Recommendation." In Proceedings of the 19th
        International Conference on World Wide Web (WWW), 2010.
    '''

    def __init__(self, actions, name="lin-ucb", recommendation_cls=None, context_size=None, alpha=0.5):
        Agent.__init__(self, name, actions)
        self.alpha = alpha
        self.context_size = context_size
        self.prev_context = None
        if context_size is not None:
            self._init_action_model()

    def _init_action_model(self):
        '''
        Summary:
            Initializes model parameters
        '''
        self.model = {'act': {}, 'act_inv': {}, 'theta': {}, 'b': {}}
        for action_id in xrange(len(self.actions)):
            self.model['act'][action_id] = np.identity(self.context_size)
            self.model['act_inv'][action_id] = np.identity(self.context_size)
            self.model['theta'][action_id] = np.zeros((self.context_size, 1))
            self.model['b'][action_id] = np.zeros((self.context_size,1))

    def _compute_score(self, context):
        '''
        Args:
            context (list)

        Returns:
            (float)
        '''

        a_inv = self.model['act_inv']
        theta = self.model['theta']

        estimated_reward = {}
        uncertainty = {}
        score = {}
        max_score = 0
        for action_id in xrange(len(self.actions)):
            action_context = np.reshape(context[action_id], (-1, 1))
            estimated_reward[action_id] = float(theta[action_id].T.dot(action_context))
            uncertainty[action_id] = float(self.alpha * np.sqrt(action_context.T.dot(a_inv[action_id]).dot(action_context)))
            score[action_id] = estimated_reward[action_id] + uncertainty[action_id]

        return score

    def _update(self, reward):
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
        # Compute score.
        context = self._pre_process_context(context)
        score = self._compute_score(context)

        # Compute best action.
        best_action = np.random.choice(self.actions)
        max_score = 0.0
        for action_id in xrange(len(self.actions)):
            if score[action_id] > max_score:
                max_score = score[action_id]
                best_action = self.actions[action_id]

        # Update
        if self.prev_action is not None:
            self._update(reward)
        self.prev_action = best_action
        self.prev_context = context
        
        return best_action

    def _pre_process_context(self, context):
        if not hasattr(context[0], '__iter__'):
            # If we only have a single context.
            new_context = {}
            for action_id in xrange(len(self.actions)):
                new_context[action_id] = context
            context = new_context

        if self.context_size is None:
            self.context_size = len(context[0])
            self._init_action_model()

        return context


