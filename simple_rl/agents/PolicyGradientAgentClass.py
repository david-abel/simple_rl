'''
PolicyGradientAgentClass.py: Class for a basic PolicyGradientAgent (one step actor critic) from:
    Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction.
    Second edition, Section 13.5. MIT Press.
'''

# Python imports.
import numpy
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent

class PolicyGradientAgent(Agent):
    ''' Implementation for a Policy Gradient Agent '''

    def __init__(self, actions, name="policy_gradient", alpha=0.1, gamma=0.99):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        '''
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha = alpha
        self.step_number = 0
        self.discount = 1.0
        self.default_pref = 0
        self.default_v = 0

        # Policy:
        self.pref_func = defaultdict(lambda : defaultdict(lambda: self.default_pref))
        # Key: state
        # Val: dict
            #   Key: action
            #   Val: preference

        # Value Function:
        self.v_func = defaultdict(lambda: self.default_v)
        # Key: state
        # Val: value

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)

        param_dict["alpha"] = self.alpha
        param_dict["gamma"] = self.gamma

        return param_dict

    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward, learning=True):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
            (str)

        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates given (s=self.prev_state,
            a=self.prev_action, r=reward, s'=state)
        '''
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)

        action = self.soft_max_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        return action

    def soft_max_policy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.

        Returns:
            (str): action.
        '''
        return numpy.random.choice(self.actions, 1, p=self.get_action_distr(state))[0]

    # -------------------------------
    # ---- POLICY AND PARAMETERS ----
    # -------------------------------

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Value Function with the TD error, then
            performs a step of policy gradient weighted by that same error.
        '''
        # If this is the first state, just return.
        if state is None:
            self.prev_state = next_state
            return

        # If the episode already ended in this state, just return.
        if state.is_terminal():
            return

        # Update the Value Function.
        td_error = reward + self.gamma * self.v_func[next_state] - self.v_func[state]
        self.v_func[state] = self.v_func[state] + self.alpha * td_error

        # Update the Policy.
        self._policy_gradient_step(state, action, self.discount * td_error)
        self.discount *= self.gamma

    def _policy_gradient_step(self, state, action, weight):
        '''
        Args:
            state (State)
            action (str)
            weight (float): How much better (or worse) than expected @action turned out.

        Summary:
            Raises the preference for @action in @state when @weight is
            positive and lowers it when @weight is negative.
        '''
        action_distr = self.get_action_distr(state)
        for i, a in enumerate(self.actions):
            grad_log_prob = (1.0 if a == action else 0.0) - action_distr[i]
            self.pref_func[state][a] = self.pref_func[state][a] + self.alpha * weight * grad_log_prob

    def get_action_distr(self, state):
        '''
        Args:
            state (State)

        Returns:
            (list of floats): The i-th float corresponds to the probability
            mass associated with the i-th action (indexing into self.actions)
        '''
        all_prefs = []
        for i, action in enumerate(self.actions):
            all_prefs.append(self.pref_func[state][action])

        # Softmax distribution.
        max_pref = max(all_prefs)
        total = sum([numpy.exp(pref - max_pref) for pref in all_prefs])
        softmax = [numpy.exp(pref - max_pref) / total for pref in all_prefs]

        return softmax

    def reset(self):
        self.step_number = 0
        self.episode_number = 0
        self.discount = 1.0
        self.pref_func = defaultdict(lambda : defaultdict(lambda: self.default_pref))
        self.v_func = defaultdict(lambda: self.default_v)
        Agent.reset(self)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        self.discount = 1.0
        Agent.end_of_episode(self)