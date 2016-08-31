'''
GradientBoostingAgentClass.py

Implementation for a Q Learner with Gradient Boosting for an approximator.

From:
    Abel, D., Agarwal, A., Diaz, F., Krishnamurthy, A., & Schapire, R. E.
    (2016). Exploratory Gradient Boosting for Reinforcement Learning in Complex Domains.
    ICML Workshop on RL and Abstraction (2016). arXiv preprint arXiv:1603.04119.
'''

# Python imports.
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# simple_rl classes.
from simple_rl.agents.QLearnerAgentClass import QLearnerAgent

class GradientBoostingAgent(QLearnerAgent):
    '''
    QLearnerAgent that uses gradient boosting with additive regression trees to approximate the Q Function.
    '''

    def __init__(self, actions, name="grad_boost", gamma=0.95, explore="softmax"):
        QLearnerAgent.__init__(self, actions=actions, name=name, gamma=gamma, explore=explore)
        self.weak_learners = []
        self.most_recent_episode = []

        # Other options to add:
            # Regressor per action
            # Tree depth as a parameter
            # Num trees per episode as a parameter.

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
        if None not in [state, action, reward, next_state]:
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
        
        # Compute Q(s,a)
        predictions = [h.predict(list(state.features()) + [self.actions.index(action)])[0] for h in self.weak_learners]
        result = float(sum(predictions)) # Cast since we'll normally get a numpy float.
        
        return result

    def add_new_weak_learner(self):
        '''
        Summary:
            Adds a new function, h, to self.weak_learners by solving for Eq. 1 using multiple additive regression trees:

            [Eq. 1] h = argmin_h (sum_i Q_A(s_i,a_i) + h(s_i, a_i) - (r_i + max_b Q_A(s'_i, b)))

        '''
        if len(self.most_recent_episode) == 0:
            # If we spawned next to the goal/died, don't add anything.
            return

        # Build up data sets of features and loss terms
        X = []
        total_loss = []

        for experience in self.most_recent_episode:
            s, a, r, s_prime = experience
            features = list(s.features()) + [self.actions.index(a)]
            loss = (r + self.gamma * self.get_max_q_value(s_prime) - self.get_q_value(s, a))
            X.append(features)
            total_loss.append(loss)

        X = np.array(X)
        
        # Compute new regressor and add it to the weak learners.
        estimator = GradientBoostingRegressor(loss='ls', n_estimators=1, max_depth=5)
        estimator.fit(X, total_loss)
        self.weak_learners.append(estimator)

    def end_of_episode(self):
        '''
        Summary:
            Performs miscellaneous end of episode tasks (printing out useful information, saving stuff, etc.)
        '''
        self.add_new_weak_learner()
        self.most_recent_episode = []
        QLearnerAgent.end_of_episode(self)
