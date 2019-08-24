class Planner(object):
    ''' Abstract class for a Planner. '''

    def __init__(self, mdp, name="planner", init_default_fields=True):

        self.name = name

        # MDP components. (Does not always apply)
        if init_default_fields:
            self.mdp = mdp
            self.init_state = self.mdp.get_init_state()
            self.states = set([])
            self.actions = mdp.get_actions()
            self.reward_func = mdp.get_reward_func()
            self.transition_func = mdp.get_transition_func()
            self.gamma = mdp.gamma
            self.has_planned = False

        def plan(self, state):
            pass

        def policy(self, state):
            pass

        def __str__(self):
            return self.name

    def plan_and_execute_next_action(self):
        """This function is supposed to plan an action, execute that action,
        and update the belief"""
        raise NotImplemented

    def execute_next_action(self, action):
        """Execute the given action, and update the belief; Useful
        for cases where the user provide's the action for debugging."""
        raise NotImplemented
