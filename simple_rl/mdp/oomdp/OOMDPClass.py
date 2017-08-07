'''
OOMDPClass.py: Implementation for Object-Oriented MDPs.

From:
	Diuk, Carlos, Andre Cohen, and Michael L. Littman.
	"An object-oriented representation for efficient reinforcement learning."
	Proceedings of the 25th international conference on Machine learning. ACM, 2008.
'''

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject

class OOMDP(MDP):
    ''' Abstract class for an Object Oriented Markov Decision Process. '''
    
    def __init__(self, actions, transition_func, reward_func, init_state, gamma=0.99):
        MDP.__init__(self, actions, transition_func, reward_func, init_state=init_state, gamma=gamma)

    def _make_oomdp_objs_from_list_of_dict(self, list_of_attr_dicts, name):
        '''
        Ags:
            list_of_attr_dicts (list of dict)
            name (str): Class of the object.

        Returns:
            (list of OOMDPObject)
        '''
        objects = []

        for attr_dict in list_of_attr_dicts:
            next_obj = OOMDPObject(attributes=attr_dict, name=name)
            objects.append(next_obj)

        return objects