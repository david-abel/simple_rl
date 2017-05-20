''' TaxiStateClass.py: Contains the TaxiState class. '''

# Local imports.
from ...mdp.oomdp.OOMDPStateClass import OOMDPState

class TaxiState(OOMDPState):
    ''' Class for Taxi World States '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]

    def __hash__(self):
    	ax = self.get_agent_x()
    	ay = self.get_agent_y()

    	state_hash = str(ax) + str(ay) + "0"

    	passengers = []

    	for p in self.objects["passenger"]:
    		state_hash += str(p["x"]) + str(p["y"]) + "0" + str(p["in_taxi"]) + "0" + str(p["dest_x"]) + str(p["dest_y"])

    	return int(state_hash)
