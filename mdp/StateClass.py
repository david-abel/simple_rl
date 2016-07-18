''' StateClass.py: Contains the State Class. '''

class State(object):
    ''' Abstract State class '''

    def __init__(self, data):
        self.data = data

    def is_terminal(self):
    	return False

    def __str__(self):
        return "s." + str(self.data)
