class State(object):
	''' Abstract State class '''
	
	def __init__(self, data):
		self.data = data

	def __str__(self):
		return "s." + str(self.data)