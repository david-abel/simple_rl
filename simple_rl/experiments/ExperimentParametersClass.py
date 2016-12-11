'''
ExperimentParametersClass.py: Contains the ExperimentParameters Class.

Purpose: Bundles all relevant parameters into an object that can be written to a file.
'''

# Python imports.
from collections import defaultdict

class ExperimentParameters(object):
    '''
    Parameters object given to @ExperimentClass instances.
    Used for storing all relevant experiment info for reproducibility.
    '''

    def __init__(self, params=defaultdict(lambda: None)):
        self.params = params

    def __str__(self):
        '''
        Summary:
            Creates a str where each key-value (parameterName-value)
            appears on a line.
        '''
        result = ""
        for item in ["\n\t"+ str(key) + " : " + str(value) for key, value in self.params.items()]:
            result += item
        return result
