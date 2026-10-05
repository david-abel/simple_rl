''' ReinforceAgentClass.py: Class for a REINFORCE agent, from:

    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
'''

# Other imports
from simple_rl.agents.PolicyGradientAgentClass import PolicyGradientAgent

class ReinforceAgent(PolicyGradientAgent):
    ''' Class for REINFORCE agent. '''

    def __init__(self, actions, name="reinforce", alpha=0.1, gamma=0.99):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        '''
        PolicyGradientAgent.__init__(self, actions, name=name, alpha=alpha, gamma=gamma)

        # Experiences from the current episode.
        self.episode = []
        # Each item: (state, action, reward)

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Stores the experience. The policy gradient steps happen in
            end_of_episode, once the return of each action is known.
        '''
        # If this is the first state just return.
        if state is None:
            self.prev_state = next_state
            return

        # If the episode already ended in this state just return.
        if state.is_terminal():
            return

        self.episode.append((state, action, reward))

    def end_of_episode(self):
        '''
        Summary:
            Performs a step of policy gradient for each action taken this
            episode, weighted by the return that followed it.
        '''
        returns = self.get_returns()

        for t, (state, action, reward) in enumerate(self.episode):
            self._policy_gradient_step(state, action, (self.gamma ** t) * returns[t])

        self.episode = []
        PolicyGradientAgent.end_of_episode(self)

    def reset(self):
        self.episode = []
        PolicyGradientAgent.reset(self)

    # ---- REINFORCE NEW ----

    def get_returns(self):
        '''
        Returns:
            (list of floats): The t-th float is the discounted sum of the
            rewards from step t of the episode onward.
        '''
        returns = [0.0] * len(self.episode)
        return_so_far = 0.0

        # Work backward, since each return builds on the one after it.
        for t in reversed(range(len(self.episode))):
            reward = self.episode[t][2]
            return_so_far = reward + self.gamma * return_so_far
            returns[t] = return_so_far

        return returns