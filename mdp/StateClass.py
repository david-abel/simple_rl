''' StateClass.py: Contains the State Class. '''

class State(object):
    ''' Abstract State class '''

    def __init__(self, data):
        self.data = data

    def features(self):
    	'''
    	Summary
    		Used by function approximators to represent the state.
    		Override this method in State subclasses to have functiona
    		approximators use a different set of features.
    	'''
    	return self.data

    def is_terminal(self):
    	return False

    def __str__(self):
        return "s." + str(self.data)

    def __eq__(self, other):
        return isinstance(other, State) and self.data == other.data
