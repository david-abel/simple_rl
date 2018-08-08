from simple_rl.mdp.MDPClass import MDP
from simple_rl.pomdp.POMDPClass import POMDP
from simple_rl.pomdp.BeliefStateClass import BeliefState

class BeliefMDP(MDP):
    def __init__(self, pomdp):
        '''
        Convert given POMDP to a Belief State MDP
        Args:
            pomdp (POMDP)
        '''
        self.state_transition_func = pomdp.transition_func
        self.state_reward_func = pomdp.reward_func
        self.state_observation_func = pomdp.observation_func
        self.belief_updater_func = pomdp.belief_updater_func

        self.pomdp = pomdp

        MDP.__init__(self, pomdp.actions, self._belief_transition_function, self._belief_reward_function,
                     BeliefState(pomdp.init_belief), pomdp.gamma, pomdp.step_cost)

    def _belief_transition_function(self, belief_state, action):
        '''
        The belief MDP transition function T(b, a) --> b' is a generative function that given a belief state and an
        action taken from that belief state, returns the most likely next belief state
        Args:
            belief_state (BeliefState)
            action (str)

        Returns:
            new_belief (defaultdict)
        '''
        observation = self._get_observation_from_environment(action)
        next_belief_distribution = self.belief_updater_func(belief_state.distribution, action, observation)
        return BeliefState(next_belief_distribution)

    def _belief_reward_function(self, belief_state, action):
        '''
        The belief MDP reward function R(b, a) is the expected reward from the POMDP reward function
        over the belief state distribution.
        Args:
            belief_state (BeliefState)
            action (str)

        Returns:
            reward (float)
        '''
        belief = belief_state.distribution
        reward = 0.
        for state in belief:
            reward += belief[state] * self.state_reward_func(state, action)
        return reward

    def _get_observation_from_environment(self, action):
        '''
        Args:
            action (str)

        Returns:
            observation (str): retrieve observation from underlying unobserved state in the POMDP
        '''
        return self.state_observation_func(self.pomdp.cur_state, action)
    
    def execute_agent_action(self, action):
        reward, next_state = super(BeliefMDP, self).execute_agent_action(action)
        self.pomdp.execute_agent_action(action)

        return reward, next_state

    def is_in_goal_state(self):
        return self.pomdp.is_in_goal_state()

if __name__ == '__main__':
    from simple_rl.tasks.maze_1d.Maze1DPOMDPClass import Maze1DPOMDP
    maze_pomdp = Maze1DPOMDP()
    maze_belief_mdp = BeliefMDP(maze_pomdp)
    maze_belief_mdp.execute_agent_action('east')
    maze_belief_mdp.execute_agent_action('east')