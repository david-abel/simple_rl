
from simple_rl.pomdp.POMDPClass import POMDP

class OOPOMDP(POMDP):

    def __init__(self, objects, classes,
                 actions, transition_func, reward_func,
                 observation_func, init_belief, init_true_state,
                 gamma=0.99, step_cost=0):
        """
        objects: a list of objects, each of a particular class.
                 The objects must be hashable.
        classes: a list of classes; classes[i] is the class for objects[i].
                 (This is useful if you want to use simple integers to represent
                 the objects.) TODO: ???
        """
        self.objects = objects
        self.classes = classes
        super().__init__(actions, transition_func, reward_func, observation_func,
                         init_belief, init_true_state, gamma=gamma, step_cost=step_cost)

    def is_in_goal_state(self):
        raise NotImplemented

    def update_belief(self):
        raise NotImplemented

    def execute_agent_action(self):
        raise NotImplemented        
