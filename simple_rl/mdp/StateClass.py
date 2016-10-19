# Python imports
import numpy

''' StateClass.py: Contains the State Class. '''

class State(object):
    ''' Abstract State class '''

    def __init__(self, data):
        self.data = data
        self._is_terminal = False

    def features(self):
    	'''
    	Summary
    		Used by function approximators to represent the state.
    		Override this method in State subclasses to have functiona
    		approximators use a different set of features.

        Returns:
            (iterable)
    	'''
        return numpy.array(self.data) # if hasattr(self.data, '__iter__') else [self.data]

    def is_terminal(self):
    	return self._is_terminal

    def set_terminal(self, is_term):
        self._is_terminal = is_term

    def __str__(self):
        return "s." + str(self.data)

    def __eq__(self, other):
        return isinstance(other, State) and self.data == other.data

    def __hash__(self):
        d = tuple(self.data)
        return hash(d)
