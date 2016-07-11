class ExperimentParameters(object):
	'''
	Parameters object given to @ExperimentClass instances.
	Used for storing all relevant experiment info for reproducibility.
	'''

	def __init__(self, params={}):
		self.params = params

	def __str__( self ):
		'''
		Summary:
			Creates a str where each key-value (parameterName-value)
			appears on a line.
		'''
		result = ""
		for item in ["\n\t"+ str(key) + " : " + str(value) for key,value in self.params.items()]:
			result += item
		return result