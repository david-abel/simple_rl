class CleanUpTask:
    def __init__(self, block_color, goal_room_color, block_name=None, goal_room_name=None):
        '''
        You can choose which attributes you would like to have represent the blocks and the rooms
        '''
        self.goal_room_name = goal_room_name
        self.block_color = block_color
        self.goal_room_color = goal_room_color
        self.block_name = block_name

    def __str__(self):
        if self.goal_room_name is None and self.block_name is None:
            return self.block_color + " to the " + self.goal_room_color + " room"
        elif self.block_name is None:
            return self.block_color + " to the room named " + self.goal_room_name
        elif self.goal_room_name is None:
            return "The block named " + self.block_name + " to the " + self.goal_room_color + " room"
        else:
            return "The block named " + self.block_name + " to the room named " + self.goal_room_name
