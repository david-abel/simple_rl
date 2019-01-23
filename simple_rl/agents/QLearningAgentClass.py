''' QLearningAgentClass.py: Class for a basic QLearningAgent '''

# Python imports.
import random
import numpy
import time
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent

class QLearningAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, actions, name="Q-learning", alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform", anneal=False):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
            explore (str): One of {softmax, uniform}. Denotes explore policy.
        '''
        name_ext = "-" + explore if explore != "uniform" else ""
        Agent.__init__(self, name=name + name_ext, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha, self.alpha_init = alpha, alpha
        self.epsilon, self.epsilon_init = epsilon, epsilon
        self.step_number = 0
        self.anneal = anneal
        self.default_q = 0 #1 / (1 - self.gamma)
        self.explore = explore

        # Q Function:
        self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))
        # Key: state
        # Val: dict
            #   Key: action
            #   Val: q-value


    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        
        param_dict["alpha"] = self.alpha
        param_dict["gamma"] = self.gamma
        param_dict["epsilon"] = self.epsilon_init
        param_dict["anneal"] = self.anneal
        param_dict["explore"] = self.explore

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
        
        if self.explore == "softmax":
            # Softmax exploration
            action = self.soft_max_policy(state)
        else:
            # Uniform exploration
            action = self.epsilon_greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if learning and self.anneal:
            self._anneal()

        return action

    def epsilon_greedy_q_policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): action.
        '''
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = numpy.random.choice(self.actions)

        return action

    def soft_max_policy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.

        Returns:
            (str): action.
        '''
        return numpy.random.choice(self.actions, 1, p=self.get_action_distr(state))[0]

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

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
        # If this is the first state, just return.
        if state is None:
            self.prev_state = next_state
            return

        # Update the Q Function.
        max_q_curr_state = self.get_max_q_value(next_state)
        prev_q_val = self.get_q_value(state, action)
        self.q_func[state][action] = (1 - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma*max_q_curr_state)

    def _anneal(self):
        # Taken from "Note on learning rate schedules for stochastic optimization, by Darken and Moody (Yale)":
        self.alpha = self.alpha_init / (1.0 +  (self.step_number / 1000.0)*(self.episode_number + 1) / 2000.0 )
        self.epsilon = self.epsilon_init / (1.0 + (self.step_number / 1000.0)*(self.episode_number + 1) / 2000.0 )

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

    def get_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float)
        '''
        return self.get_max_q_value(state)

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state, @action) pair.
        '''
        return self.q_func[state][action]

    def get_action_distr(self, state, beta=0.2):
        '''
        Args:
            state (State)
            beta (float): Softmax temperature parameter.

        Returns:
            (list of floats): The i-th float corresponds to the probability
            mass associated with the i-th action (indexing into self.actions)
        '''
        all_q_vals = []
        for i, action in enumerate(self.actions):
            all_q_vals.append(self.get_q_value(state, action))

        # Softmax distribution.
        total = sum([numpy.exp(beta * qv) for qv in all_q_vals])
        softmax = [numpy.exp(beta * qv) / total for qv in all_q_vals]

        return softmax

    def reset(self):
        self.step_number = 0
        self.episode_number = 0
        self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))
        Agent.reset(self)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        if self.anneal:
            self._anneal()
        Agent.end_of_episode(self)

    def print_q_func(self):
        '''
        Summary:
            Prints the Q function.
        '''
        for state in self.q_func.keys():
            print(state)
            for action in self.actions:
                print("    ", action, self.q_func[state][action])
