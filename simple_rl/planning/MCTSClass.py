''' MCTSClass.py: Class for a basic Monte Carlo Tree Search Planner. '''

# Python imports.
import math as m
import random as r
from collections import defaultdict

# Other imports.
from PlannerClass import Planner

class MCTS(Planner):

    def __init__(self, mdp, name="mcts", explore_param=m.sqrt(2), rollout_depth=100, num_rollouts_per_step=50):
        Planner.__init__(self, mdp, name=name)

        self.rollout_depth = rollout_depth
        self.num_rollouts_per_step = num_rollouts_per_step
        self.value_total = defaultdict(lambda : defaultdict(float))
        self.explore_param = explore_param
        self.visitation_counts = defaultdict(lambda : defaultdict(lambda : 1))

    def plan(self, cur_state, horizon=100):
        '''
        Args:
            cur_state (State)
            horizon (int)

        Returns:
            (list): List of actions
        '''
        action_seq = []
        state_seq = [cur_state]
        steps = 0
        while not cur_state.is_terminal() and steps < horizon:
            action = self.policy(cur_state)
            cur_state = self.transition_func(cur_state, action)
            action_seq.append(action)
            state_seq.append(cur_state)
            steps += 1

        return action_seq, state_seq

    def policy(self, state):
        for i in xrange(self.num_rollouts_per_step):
            a = self._next_action(state)
            self._rollout(state, a)

        return self._next_action(state)

    def _next_action(self, state):
        '''
        Args;
            state (State)

        Returns:
            (str)

        Summary:
            Performs a single step of the MCTS rollout.
        '''
        best_action = self.actions[0]
        best_score = 0

        total_visits = [self.visitation_counts[state][a] for a in self.actions]

        if 0 in total_visits:
            # Insufficient stats, return random.
            return random.choice(self.actions)

        total = sum(total_visits)

        # Else choose according to the UCT explore method.
        for cur_action in self.actions:
            s_a_value_tot = self.value_total[state][cur_action]
            s_a_visit = self.visitation_counts[state][cur_action]
            score = s_a_value_tot / s_a_visit + self.explore_param * m.sqrt(m.log(total) / s_a_visit)

            if score > best_score:
                best_action = cur_action
                best_score = score

        return best_action

    def _rollout(self, cur_state, action):
        trajectory = []
        total_discounted_reward = []
        for i in range(self.rollout_depth):


            # Simulate next.
            next_action = self._next_action(cur_state)
            cur_state = self.transition_func(cur_state, next_action)
            next_reward = self.transition_func(cur_state, next_action)

            # Track rewards and states.
            total_discounted_reward.append(self.gamma**i * next_reward)
            trajectory.append((cur_state, next_action))

            if cur_state.is_terminal():
                # Break terminal.
                break

        # Update all visited nodes.
        for i, experience in enumerate(trajectory):
            s, a = experience
            self.visitation_counts[s][a] += 1
            self.value_total[s][a] += sum(total_discounted_reward[i:])

        return total_discounted_reward

