from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState

class TrenchOOMDPState(OOMDPState):
    ''' Class for Trench World States '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def __hash__(self):
        state_hash = str(self.get_agent_x()) + str(self.get_agent_y()) + str(self.objects["agent"][0]["dx"] + 1)\
                     + str(self.objects["agent"][0]["dy"] + 1) + str(self.objects["agent"][0]["dest_x"])\
                     + str(self.objects["agent"][0]["dest_x"]) + str(self.objects["agent"][0]["dest_y"]) + \
                     str(self.objects["agent"][0]["has_block"]) + "00"

        for b in self.objects["block"]:
            state_hash += str(b["x"]) + str(b["y"])

        state_hash += "00"

        for l in self.objects["lava"]:
            state_hash += str(l["x"]) + str(l["y"])

        return int(state_hash)

    def __eq__(self, other_trench_state):
        return hash(self) == hash(other_trench_state)
