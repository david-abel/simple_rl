# Python imports.
from collections import defaultdict
import Queue
import random

# Other impotrs.
from simple_rl.tasks import ChainMDP, GridWorldMDP

class ValueIteration(object):

    def __init__(self, mdp, delta=0.01, max_iterations=100, sample_rate=1):
        '''
        Args:
            mdp (MDP)
            delta (float): After an iteration if VI, if no change more than @\delta has occurred, terminates.
            max_iterations (int): Hard limit for number of iterations.
            sample_rate (int): Determines how many samples from @mdp to take to estimate T(s' | s, a).
        '''
        self.delta = delta
        self.max_iterations = max_iterations
        self.sample_rate = sample_rate
        self.value_func = defaultdict(float)

        # MDP components.
        self.mdp = mdp
        self.init_state = self.mdp.get_init_state()
        self.S = []
        self.A = mdp.actions
        self.R = mdp.reward_func
        self.T = mdp.transition_func
        self.gamma = mdp.gamma
        self._compute_reachable_state_space()

    def get_num_states(self):
        return len(self.S)      

    def get_states(self):
        return self.S

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
            next_state = self.T(s,a)
            next_state_counts[next_state] += 1
            
        # Compute T(s' | s, a) estimate based on MLE.
        next_state_probs = defaultdict(float)
        for key in next_state_counts:
            next_state_probs[key] = float(next_state_counts[key]) / self.sample_rate

        # Compute expected value.
        expected_future_val = 0
        for key in next_state_probs:
            expected_future_val += next_state_probs[key] * self.value_func[next_state]

        return self.R(s,a) + self.gamma*expected_future_val

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states.
        '''
        state_queue = Queue.Queue()
        state_queue.put(self.init_state)
        self.S.append(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.A:
                for samples in xrange(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    next_state = self.T(s,a)

                    # if s.objects["agent"][0]["x"] == s.objects["passenger"][0]["x"] \
                    # and s.objects["agent"][0]["y"] == s.objects["passenger"][0]["y"]:
                    #   print s, a, next_state

                    if next_state not in self.S:
                        self.S.append(next_state)
                        state_queue.put(next_state)

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
            for s in self.S:
                
                if s.is_terminal():
                    continue

                max_q = float("-inf")
                for a in self.A:
                    q_s_a = self.get_q_value(s, a)
                    max_q = q_s_a if q_s_a > max_q else max_q

                # Check terminating condition.
                diff = abs(self.value_func[s] - max_q)
                max_diff = diff if diff > max_diff else max_diff

                # Update value.
                self.value_func[s] = max_q
            iterations += 1

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
        plan = []
        state_seq = [state]
        steps = 0
        while (not state.is_terminal()) and steps < horizon:
            next_action = self._get_max_q_action(state)
            plan.append(next_action)
            state = self.T(state, next_action)
            state_seq.append(state)
            steps += 1

        return plan, state_seq
    
    def _get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.A)
        max_q_val = float("-inf")
        shuffled_action_list = self.A[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action


def main():
    # mdp = GridWorldMDP()
    mdp = ChainMDP()
    vi = ValueIteration(mdp)
    vi.run_vi()

    vi.print_value_func()

if __name__ == "__main__":
    main()