''' TaxiStateClass.py: Contains the TaxiState class. '''

# Local libs.
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState

class TaxiState(OOMDPState):
    ''' Class for Taxi World States '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]
