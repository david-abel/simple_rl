''' OOMDPStateClass.py: Contains the OOMDP State Class. '''

# Local libs.
from simple_rl.mdp.StateClass import State

class OOMDPState(State):
    ''' OOMDP State class '''

    def __init__(self, objects):
        '''
        Args:
            objects (dict of OOMDPObject instances): {key=object class (str):val = object instances}
        '''
        self.objects = objects

        # Turn object attributes into a feature list.
        state_vec = []
        for obj_class in self.objects.keys():
            for obj in self.objects[obj_class]:
              state_vec += [obj.get_obj_state()]

        State.__init__(self, data=state_vec)

    def __str__(self):
        result = ""
        for obj_class in self.objects.keys():
            for obj in self.objects[obj_class]:
                result += "\t" + str(obj)
            result += "\n"
        return result
