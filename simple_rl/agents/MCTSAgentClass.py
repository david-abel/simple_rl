''' MCTSAgentClass.py: Class for a basic Monte Carlo Tree Search '''

# Python imports.
import math as m
import random as r
from collections import defaultdict

# Local imports.
from AgentClass import Agent
from simple_rl.tasks.gym.GymStateClass import GymState

class MCTSAgent(Agent):

    def __init__(self, actions, env_model, explore_param=m.sqrt(2), rollout_depth=100, num_rollouts_per_step=50, name="mcts", gamma=0.99):
        self.env_model = env_model
        self.rollout_depth = rollout_depth
        self.num_rollouts_per_step = num_rollouts_per_step
        self.value_total = defaultdict(float)
        self.explore_param = explore_param
        self.visitation_counts = defaultdict(lambda : 1)

        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

    def act(self, state, reward):
        # value_estimate = {k : None for k in self.actions}

        for i in xrange(self.num_rollouts_per_step):
            # for a in self.actions:
            a = self._next_action(state)
            self._rollout(state, a)

        return self._next_action(state)

    def _next_action(self, state):
        best_action = self.actions[0]
        best_score = 0

        total_visits = [self.visitation_counts[(state, a)] for a in self.actions]

        if 0 in total_visits:
            # Insufficient stats, return random.
            return random.choice(self.actions)

        t = sum(total_visits)

        # Else choose according to the UCT explore method.
        for cur_action in self.actions:
            w = self.value_total[(state, cur_action)]
            n = self.visitation_counts[(state, cur_action)]
            score = w / n + self.explore_param * m.sqrt(m.log(t) / n)

            if score > best_score:
                best_action = cur_action
                best_score = score

        return best_action

    def _rollout(self, state, action):
        next_state = state #self.t_model(state, action)

        trajectory = [] # 

        total_discounted_reward = []
        for i in range(self.rollout_depth):

            next_action = self._next_action(state) #self._next_action(state)

            # Simulate next.
            next_obs, next_reward, is_terminal, _ = self.env_model._step(next_action)
            next_state = GymState(next_obs)

            # Track rewards and states.
            total_discounted_reward.append(self.gamma**i * next_reward)
            trajectory.append((next_state, next_action))

            if is_terminal:
                break

        self.env_model._set_current_observation(state.data)

        # Update all visited nodes.
        for i, experience in enumerate(trajectory):
            s, a = experience
            self.visitation_counts[(s, a)] += 1
            self.value_total[(s, a)] += sum(total_discounted_reward[i:])

        return total_discounted_reward


