# OOPOMDP implementation

from simple_rl.pomdp.POMDPClass import POMDP
from simple_rl.mdp.StateClass import State
from moos3d.planning.shared import BeliefState
import pprint
import copy

class OOPOMDP_ObjectState(State):
    def __init__(self, objclass, attributes):
        """
        class: "class",
        attributes: {
            "attr1": value,
            ...
        }
        """
        self.objclass = objclass
        self.attributes = attributes
        self._to_hash = pprint.pformat(self.attributes)   # automatically sorted by keys

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'OOPOMDP_ObjectState::(%s,%s)' % (str(self.objclass),
                                                 str(self.attributes))
    
    def __hash__(self):
        return hash(self._to_hash)

    def __eq__(self, other):
        return self.objclass == other.objclass\
            and self.attributes == other.attributes

    def __getitem__(self, attr):
        return self.attributes[attr]

    def __setitem__(self, attr, value):
        self.attributes[attr] = value
    
    def __len__(self):
        return len(self.attributes)

    def copy(self):
        return OOPOMDP_ObjectState(self.objclass, copy.deepcopy(self.attributes))
    

class OOPOMDP_State(State):

    def __init__(self, object_states):
        """
        objects_states: dictionary of dictionaries; Each dictionary represents an object state:
            { ID: ObjectState }
        """
        # internally, objects are sorted by ID.
        self.object_states = object_states
        self._to_hash = pprint.pformat(self.object_states)  # automatically sorted by keys
        super().__init__(object_states)

    def __str__(self):
        return 'OOPOMDP_State::[%s]' % str(self.object_states)

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.object_states == other.object_states

    def __hash__(self):
        return hash(self._to_hash)

    def __getitem__(self, index):
        raise NotImplemented
    
    def __len__(self):
        raise NotImplemented

    def get_object_state(self, objid):
        return self.object_states[objid]

    def set_object_state(self, objid, object_state):
        self.object_states[objid] = object_state

    def get_object_class(self, objid):
        return self.object_states[objid].objclass

    def get_object_attribute(self, objid, attr):
        return self.object_states[objid][attr]

    def copy(self):
        return OOPOMDP_State(copy.deepcopy(self.object_states))


class OOPOMDP_BeliefState(BeliefState):
    """
    The belief state in an OOPOMDP is a collection of belief states
    for all objects."""
    def __init__(self, belief_states):
        """
        belief_states: dictionary, map from object ID to a BeliefState.
        """
        self._belief_states = belief_states
        self.distribution = None  # there's no need to represent the entire distribution

    def get_distribution(self, object_id):
        return self._belief_states[object_id].distribution

    def set_distribution(self, object_id, distribution):
        self._belief_states[object_id].distribution = distribution

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'OOPOMDP_BeliefState::' + str(self._belief_states)

    def __iter__(self):
        return iter(self._belief_states)

    def sample(self, sampling_method='max'):
        '''
        Returns:
            sampled_state (State)
        '''
        if sampling_method == 'max':
            object_states = {}
            for objid in self._belief_states:
                object_states[objid] = self.get_distribution(objid).mpe()
            return OOPOMDP_State(object_states)
        if sampling_method == 'random':
            object_states = {}
            for objid in self._belief_states:
                object_states[objid] = self.get_distribution(objid).random()
            return OOPOMDP_State(object_states)            
        raise NotImplementedError('Sampling method {} not implemented yet'.format(sampling_method))

    def update(self, real_action, real_observation, **kwargs):
        robot_id = kwargs.get('robot_id', None)
        robot_state_trans_func = kwargs.get('robot_state_transition_func', None)

        # update belief of robot pose - deterministic
        robot_state = self.get_distribution(robot_id).mpe()
        next_robot_state = robot_state_trans_func(robot_state, real_action, real_observation)
        self.get_distribution(robot_id)[robot_state] = 0.0
        self.get_distribution(robot_id)[next_robot_state] = 1.0

        for objid in self._belief_states:
            if objid != robot_id:
                new_obj_distribution = self.get_distribution(objid).update(real_action, real_observation,
                                                                           robot_state=robot_state, next_robot_state=next_robot_state,
                                                                           **kwargs)
                self.set_distribution(objid, new_obj_distribution)
                mpe = new_obj_distribution.mpe()
                if hasattr(kwargs, "verbose"):
                    print("MPE for obj %d: %s : %.3f" % (objid, mpe, new_obj_distribution[mpe]))
        

class OOPOMDP(POMDP):

    def __init__(self, attributes, domains,
                 actions, transition_func, reward_func, observation_func,
                 init_belief, init_objects_state, gamma=0.99, step_cost=0):
        """
        attributes: a dictionary that maps from class to a set of strings.
        domains: a dictionary that maps from (class, attribute) to a function 
                 that takes as input the object's attribute, with bool return value.
        init_objects_state: an instance of OOPOMDP_State which contains the TRUE inital
                state of all the objects. 
        """
        self.classes = list(attributes.keys())
        self.attributes = attributes
        self.domains = domains

        super().__init__(actions, transition_func, reward_func, observation_func,
                         init_belief, init_objects_state, gamma=gamma, step_cost=step_cost)
        
    def verify_state(self, state):
        """Returns true if state (OOPOMDP_State) is valid."""
        object_states = state.object_states
        for objid in object_states:
            objclass = object_states[objid].objclass
            if objclass not in self.attributes:
                return False
            attrs = object_states[objid].attributes
            for attr in attrs:
                if attr not in self.attributes[objclass]:
                    return False
                attr_value = object_states[objid][attr]
                if not self.domains[(objclass, attr)](attr_value):
                    return False
        return True
