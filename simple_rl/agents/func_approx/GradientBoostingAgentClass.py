'''
GradientBoostingAgentClass.py

Implementation for a Q Learner with Gradient Boosting for an approximator.

From:
    Abel, D., Agarwal, A., Diaz, F., Krishnamurthy, A., & Schapire, R. E.
    (2016). Exploratory Gradient Boosting for Reinforcement Learning in Complex Domains.
    ICML Workshop on RL and Abstraction (2016). arXiv pre#print arXiv:1603.04119.
'''

# Python imports.
import random
import math
import numpy as np
import time
try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    print "Error: sklearn not installed. See: http://scikit-learn.org/stable/install.html"
    quit()

# simple_rl classes.
from ..QLearnerAgentClass import QLearnerAgent

class GradientBoostingAgent(QLearnerAgent):
    '''
    QLearnerAgent that uses gradient boosting with additive regression trees to approximate the Q Function.
    '''

    def __init__(self, actions, name="grad_boost", gamma=0.99, explore="softmax", markov_window=20, update_window=500):
        name += "-m" if markov_window > 0 else ""
        QLearnerAgent.__init__(self, actions=actions, name=name, gamma=gamma, explore=explore)
        self.weak_learners = []
        self.model = []
        self.most_recent_episode = []
        self.max_state_features = 0
        self.max_depth = len(actions)*2
        self.markov_window = markov_window
        self.update_window = 500

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''

        # Update on a per step basis.
        if self.step_number > 0 and self.step_number % self.update_window == 0:
            self.add_new_weak_learner()
            self.most_recent_episode = []

            if self.markov_window > 0:
                self.model = self.weak_learners[-self.markov_window:]
            else:
                self.model = self.weak_learners

        if None not in [state, action, reward, next_state]:
            if len(state.features()) > self.max_state_features:
                self.max_state_features = len(state.features())
            self.most_recent_episode.append((state, action, reward, next_state))

    def get_q_value(self, state, action):
        '''
        Args:
            state (State): A State object containing the abstract state representation
            action (str): A string representing an action. See namespaceAIX.

        Summary:
            Retrieves the Q Value associated with this state/action pair. Computed via summing h functions.

        Returns:
            (float): denoting the q value of the (@state,@action) pair.
        '''
        if len(self.weak_learners) == 0:
            # Default Q value.
            return 0

        features = self._pad_features_with_zeros(state, action)

        # Compute Q(s,a)
        predictions = [h.predict(features)[0] for h in self.model]
        result = float(sum(predictions)) # Cast since we'll normally get a numpy float.
        
        return result

    def _pad_features_with_zeros(self, state, action):
        '''
        Args:
            features (iterable)

        Returns:
            (list): Of the same length as self.max_state_features
        '''
        features = state.features()
        while len(features) < self.max_state_features:
            features = np.append(features, 0)

        # Reshape per update to cluster regression in sklearn 0.17.
        reshaped_features = np.append(features, [self.actions.index(action)])
        reshaped_features = reshaped_features.reshape(1, -1)

        return reshaped_features

    def add_new_weak_learner(self):
        '''
        Summary:
            Adds a new function, h, to self.weak_learners by solving for Eq. 1 using multiple additive regression trees:

            [Eq. 1] h = argmin_h (sum_i Q_A(s_i,a_i) + h(s_i, a_i) - (r_i + max_b Q_A(s'_i, b)))

        '''
        if len(self.most_recent_episode) == 0:
            # If this episode contains no data, don't do anything.
            return

        # Build up data sets of features and loss terms
        data = np.zeros((len(self.most_recent_episode), self.max_state_features + 1))
        total_loss = np.zeros(len(self.most_recent_episode))

        for i, experience in enumerate(self.most_recent_episode):
            # Grab the experience.
            s, a, r, s_prime = experience

            # Pad in case the state features are too short (as in Atari sometimes).
            features = self._pad_features_with_zeros(s, a)
            loss = (r + self.gamma * self.get_max_q_value(s_prime) - self.get_q_value(s, a))
            
            # Add to relevant lists.
            data[i] = features
            total_loss[i] = loss

        # Compute new regressor and add it to the weak learners.
        estimator = GradientBoostingRegressor(loss='ls', n_estimators=1, max_depth=self.max_depth)
        estimator.fit(data, total_loss)
        self.weak_learners.append(estimator)

    def end_of_episode(self):
        '''
        Summary:
            Performs miscellaneous end of episode tasks (#printing out useful information, saving stuff, etc.)
        '''

        # self.model = self.weak_learners
        self.add_new_weak_learner()
        self.most_recent_episode = []

        if self.markov_window > 0:
            # num_sampled_trees = int(math.ceil(len(self.weak_learners) / 10.0))
            # self.model = random.sample(self.weak_learners, num_sampled_trees)
            self.model = self.weak_learners[-self.markov_window:]
        else:
            self.model = self.weak_learners

        Agent.end_of_episode(self)
