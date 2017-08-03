# Python imports.
from collections import defaultdict
import Queue
import random

# Other imports.
from PlannerClass import Planner

class ValueIteration(Planner):

    def __init__(self, mdp, name="value_iter", delta=0.0001, max_iterations=500, sample_rate=1):
        '''
        Args:
            mdp (MDP)
            delta (float): After an iteration if VI, if no change more than @\delta has occurred, terminates.
            max_iterations (int): Hard limit for number of iterations.
            sample_rate (int): Determines how many samples from @mdp to take to estimate T(s' | s, a).
        '''
        Planner.__init__(self, mdp, name=name)

        self.delta = delta
        self.max_iterations = max_iterations
        self.sample_rate = sample_rate
        self.value_func = defaultdict(float)
        self.reachability_done = False
        self._compute_reachable_state_space()

    def get_num_states(self):
        return len(self.states)      

    def get_states(self):
        if self.reachability_done:
            return self.states
        else:
            self._compute_reachable_state_space()
            return self.states

    def get_value(self, s):
        '''
        Args:
            s (State)

        Returns:
            (float)
        '''
        return self._compute_max_qval_action_pair(s)[0]

    def get_q_value(self, s, a):
        '''
        Args:
            s (State)
            a (str): action

        Returns:
            (float): The Q estimate given the current value function @self.value_func.
        '''
        
        # Take samples and track next state counts.
        next_state_counts = defaultdict(int)
        for samples in xrange(self.sample_rate): # Take @sample_rate samples to estimate E[V]
            next_state = self.transition_func(s,a)
            next_state_counts[next_state] += 1

        # Compute T(s' | s, a) estimate based on MLE.
        next_state_probs = defaultdict(float)
        for state in next_state_counts:
            next_state_probs[state] = float(next_state_counts[state]) / self.sample_rate

        # Compute expected value.
        expected_future_val = 0
        for state in next_state_probs:
            expected_future_val += next_state_probs[state] * self.value_func[state]

        return self.reward_func(s,a) + self.gamma*expected_future_val

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''
        state_queue = Queue.Queue()
        state_queue.put(self.init_state)
        self.states.append(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in xrange(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    next_state = self.transition_func(s,a)

                    if next_state not in self.states:
                        self.states.append(next_state)
                        state_queue.put(next_state)

        self.reachability_done = True

    def run_vi(self):
        '''
        Summary:
            Runs ValueIteration and fills in the self.value_func.           
        '''
        # Algorithm bookkeeping params.
        iterations = 0
        max_diff = float("inf")

        # Main loop.
        while max_diff > self.delta and iterations < self.max_iterations:
            max_diff = 0
            for s in self.states:
                if s.is_terminal():
                    continue

                max_q = float("-inf")
                for a in self.actions:
                    q_s_a = self.get_q_value(s, a)
                    max_q = q_s_a if q_s_a > max_q else max_q
                # Check terminating condition.
                max_diff = max(abs(self.value_func[s] - max_q), max_diff)

                # Update value.
                self.value_func[s] = max_q
            iterations += 1

        value_of_init_state = self._compute_max_qval_action_pair(self.init_state)[0]
        return iterations, value_of_init_state

    def print_value_func(self):
        for key in self.value_func.keys():
            print key, ":", self.value_func[key]

    def plan(self, state, horizon=100):
        '''
        Args:
            state (State)
            horizon (int)

        Returns:
            (list): List of actions
        '''
        action_seq = []
        state_seq = [state]
        steps = 0

        while (not state.is_terminal()) and steps < horizon:
            next_action = self._get_max_q_action(state)
            action_seq.append(next_action)
            state = self.transition_func(state, next_action)
            state_seq.append(state)
            steps += 1

        return action_seq, state_seq
    
    def _get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): Action

        Summary:
            For use in a FixedPolicyAgent.
        '''
        return self._get_max_q_action(state)

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
