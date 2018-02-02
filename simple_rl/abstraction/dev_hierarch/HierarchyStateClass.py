from simple_rl.mdp.StateClass import State

''' HierarchyStateClass.py: Contains the HierarchyState Class. '''

class HierarchyState(State):

    def __init__(self, data=[], is_terminal=False, level=0):
        self.level = level
        State.__init__(self, data=data, is_terminal=is_terminal)

    def get_level(self):
        return self.level

    def __str__(self):
        return State.__str__(self) + "-lvl=" + str(self.level)
