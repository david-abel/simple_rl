# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.pomdp.BeliefUpdaterClass import BeliefUpdater
from simple_rl.mdp.MDPClass import MDP

class POMDP(MDP):
    ''' Abstract class for a Partially Observable Markov Decision Process. '''

    def __init__(self, actions, transition_func, reward_func, observation_func,
                 init_belief, init_true_state, belief_updater_type='discrete', gamma=0.99, step_cost=0):
        '''
        In addition to the input parameters needed to define an MDP, the POMDP
        definition requires an observation function, a way to update the belief
        state and an initial belief.
        Args:
            actions (list)
            observations (list)
            transition_func: T(s, a) -> s'
            reward_func: R(s, a) -> float
            observation_func: O(s, a) -> z
            init_belief (defaultdict): initial probability distribution over states
            belief_updater_type (str): discrete/kalman/particle
            gamma (float)
            step_cost (int)
        '''
        self.observation_func = observation_func
        self.init_belief = init_belief
        self.cur_belief = init_belief
        MDP.__init__(self, actions, transition_func, reward_func, init_true_state, gamma, step_cost)

        if belief_updater_type is not None:
            self.belief_updater = BeliefUpdater(self, transition_func, reward_func, observation_func, belief_updater_type)
            self.belief_updater_func = self.belief_updater.updater

    def get_cur_belief(self):
        return self.cur_belief

    def get_observation_func(self):
        '''
        Returns:
            observation_function: O(s, a) -> o
        '''
        return self.observation_func

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            reward (float)
            next_belief (defaultdict)
        '''
        observation = self.observation_func(self.cur_state, action)
        ### NO BELIEF UPDATE HAPPENS. Belief maintained by the Planner.
        # new_belief = self.belief_updater_func(self.cur_belief, action, observation)
        # self.cur_belief = new_belief

        reward, next_state = super(POMDP, self).execute_agent_action(action)

        return reward, observation, new_belief
