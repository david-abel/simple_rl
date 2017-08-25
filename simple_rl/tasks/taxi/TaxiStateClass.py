''' TaxiStateClass.py: Contains the TaxiState class. '''

# Other imports
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState

class TaxiState(OOMDPState):
    ''' Class for Taxi World States '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def __hash__(self):

    	state_hash = str(self.get_agent_x()) + str(self.get_agent_y()) + "00"

    	for p in self.objects["passenger"]:
    		state_hash += str(p["x"]) + str(p["y"]) + str(p["in_taxi"])

    	return int(state_hash)

    def __eq__(self, other_taxi_state):
        return hash(self) == hash(other_taxi_state)
