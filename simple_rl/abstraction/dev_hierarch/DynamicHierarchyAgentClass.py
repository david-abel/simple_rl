# Python imports.
from collections import defaultdict

# Other imports.
from HierarchyAgentClass import HierarchyAgent

class DynamicHierarchyAgent(HierarchyAgent):
    
    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        new_level = self._compute_max_v_hat_level(ground_state)
        self.set_level(new_level)
        action = HierarchyAgent.act(self, ground_state, reward)

        return action

    def _compute_max_v_hat_level(self, ground_state):
        '''
        Args:
            ground_state (simple_rl.mdp.State)

        Returns:
            (int): The level with the highest value estimate.
        '''
        max_q = 0
        best_lvl = 0
        for lvl in range(self.get_num_levels()):
            abstr_state = self.state_abstr_stack.phi(ground_state, lvl)
            v_hat = self.agent.get_max_q_value(abstr_state)

            if v_hat > max_q:
                best_lvl = lvl
                max_q = v_hat

        return best_lvl

