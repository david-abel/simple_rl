
# Other imports.
from simple_rl.agents import Agent

class HierarchyAgent(Agent):

    def __init__(self, SubAgentClass, sa_stack, aa_stack, cur_level=0, name_ext=""):
        '''
        Args:
            sa_stack (StateAbstractionStack)
            aa_stack (ActionAbstractionStack)
            cur_level (int): Must be in [0:len(state_abstr_stack)]
        '''
        # Setup the abstracted agent.
        self.state_abstr_stack = sa_stack
        self.action_abstr_stack = aa_stack
        self.cur_level = cur_level
        self.agent = SubAgentClass(actions=self.get_cur_actions())
        Agent.__init__(self, name=self.agent.name + "-hierarch" + name_ext, actions=self.get_cur_actions())

    # -- Accessors --

    def get_num_levels(self):
        return self.state_abstr_stack.get_num_levels()

    def get_cur_actions(self):
        if self.cur_level == 0:
            return self.action_abstr_stack.prim_actions

        return self.get_cur_action_abstr().get_actions()

    def get_cur_action_abstr(self):
        return self.action_abstr_stack.get_aa_list()[self.cur_level - 1]

    def get_cur_abstr_state(self, state):
        return self.state_abstr_stack.phi(state, self.cur_level)

    # -- Mutators --

    def add_sa_aa_pair(self, sa, aa):
        self.state_abstr_stack.add_sa(sa)
        self.action_abstr_stack.add_aa(aa)

    def incr_level(self):
        self.cur_level = min(self.cur_level + 1, self.state_abstr_stack.get_num_levels())

    def decr_level(self):
        self.cur_level = min(self.cur_level - 1, 0)

    def set_level(self, new_level):
        if new_level < 0 or new_level > self.get_num_levels():
            raise ValueError("HierarchyAgentError: the given level (" + str(new_level) +") exceeds the hierarchy height (" + str(self.get_num_levels()) + ")")

        self.cur_level = new_level

    # -- Central Act Method --

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        # Give the SA stack, ground state, and reward to the AA stack.
        return self.action_abstr_stack.act(self.agent, self.state_abstr_stack, ground_state, reward, level=self.cur_level)

    # -- Reset --

    def reset(self):
        self.agent.reset()
        for aa in self.action_abstr_stack.get_aa_list():
            aa.reset()

    def end_of_episode(self):
        self.agent.end_of_episode()
        for aa in self.action_abstr_stack.get_aa_list():
            aa.end_of_episode()

    def _reset_reward(self):
        self.agent._reset_reward()
