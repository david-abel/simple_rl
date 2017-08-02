''' BlockDudeStateClass.py: Contains the BlockDudeState class. '''

# Other imports
from ...mdp.oomdp.OOMDPStateClass import OOMDPState

class BlockDudeState(OOMDPState):
    ''' Class for Block Dude State '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def get_agent_dir(self):
        return self.objects["agent"][0]["facing"]

    def is_solid_object_at(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff there is a wall or block object at (@x, @y).
        '''
        objs_at_loc = []
        for obj_class in ["wall", "block"]:
            for obj_instance in self.objects[obj_class]:
                if obj_instance["x"] == x and obj_instance["y"] == "y":
                    return True

        return False

    def get_block_index_at_loc(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (int): The index of the block at (@x, @y). None if no such block.
        '''
        objs_at_loc = []
        for i, obj_instance in enumerate(self.objects["block"]):
            if obj_instance["x"] == x and obj_instance["y"] == "y":
                return i

    def get_carried_block(self):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (int): The index of the block being carried. None if no such block.
        '''
        objs_at_loc = []
        for i, obj_instance in enumerate(self.objects["block"]):
            if obj_instance["carried"]:
                return i

    def __hash__(self):
    	ax = self.get_agent_x()
    	ay = self.get_agent_y()

    	state_hash = str(ax) + str(ay) + "0"

    	for b in self.objects["block"]:
    		state_hash += str(b["x"]) + str(b["y"]) + "0" + str(b["in_taxi"])

    	return int(state_hash)
