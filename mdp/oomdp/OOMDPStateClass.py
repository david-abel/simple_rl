''' OOMDPStateClass.py: Contains the OOMDP State Class. '''

# Local libs.
from simple_rl.mdp.StateClass import State

class OOMDPState(State):
    ''' Abstract OOMDP State class '''

    def __init__(self, objects):
        '''
        Args:
            objects (dict of OOMDPObject instances): {key=object class (str):val = object instances}
        '''
        self.objects = objects
        State.__init__(self, data=self.create_vec_from_objects())

    def create_vec_from_objects(self):
        '''
        Returns:
            A vector of integers corresponding to the features for this state.
        '''
        state_vec = []
        for obj_class in self.objects.keys():
            for obj in self.objects[obj_class]:
              state_vec += [obj.get_obj_state()]
        return state_vec

    def __str__(self):
        result = ""
        for obj_class in self.objects.keys():
            for obj in self.objects[obj_class]:
                result += "\t" + str(obj)
            result += "\n"
        return result
