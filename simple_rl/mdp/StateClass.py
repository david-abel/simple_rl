# Python imports
import numpy

''' StateClass.py: Contains the State Class. '''

class State(object):
    ''' Abstract State class '''

    def __init__(self, data=[], is_terminal=False):
        self.data = data
        self._is_terminal = is_terminal

    def features(self):
    	'''
    	Summary
    		Used by function approximators to represent the state.
    		Override this method in State subclasses to have functiona
    		approximators use a different set of features.

        Returns:
            (iterable)
    	'''
        return numpy.array(self.data)

    def __hash__(self):
        return hash(self.data)

    def is_terminal(self):
    	return self._is_terminal

    def set_terminal(self, is_term):
        self._is_terminal = is_term

    def __str__(self):
        return "s." + str(self.data)

    def __eq__(self, other):
        # isinstance(other, State) and 
        return self.data == other.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
