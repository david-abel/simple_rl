class Planner(object):
    ''' Abstract class for a Planner. '''

    def __init__(self, mdp, name="planner"):

        self.name = name

        # MDP components.
        self.mdp = mdp
        self.init_state = self.mdp.get_init_state()
        self.states = []
        self.actions = mdp.actions
        self.reward_func = mdp.reward_func
        self.transition_func = mdp.transition_func
        self.gamma = mdp.gamma

        def plan(self, state):
            pass

        def policy(self, state):
            pass

        def __str__(self):
            return self.name