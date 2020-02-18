# Python import.
import copy
import random

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.cleanup.cleanup_task import CleanUpTask


class CleanUpMDP(MDP):
    ACTIONS = ["up", "down", "left", "right"]
    CLASSES = ["room", "block", "door"]  # TODO NOTE: CURRENTLY UNUSED

    # TODO NOTE MAYBE ADD AGENT CLASS.

    def __init__(self, task, init_loc=(0, 0), blocks=[], rooms=[], doors=[], rand_init=False, gamma=0.99,
                 init_state=None):
        '''
        :param task: The given CleanUpTask for this MDP
        :param init_loc: Initial agent location
        :param blocks: List of blocks
        :param rooms: List of rooms
        :param doors: List of doors
        :param rand_init: random initialization boolean
        :param gamma: gamma factor
        :param init_state: Initial state if given
        '''
        from simple_rl.tasks.cleanup.cleanup_state import CleanUpState
        self.task = task
        if rand_init:
            block_loc = [(x, y) for block in blocks for (x, y) in (block.x, block.y)]
            init_loc = random.choice(
                [(x, y) for room in rooms for (x, y) in room.points_in_room if (x, y) not in block_loc])
        init_state = CleanUpState(task, init_loc[0], init_loc[1], blocks=blocks, doors=doors, rooms=rooms) \
            if init_state is None or rand_init else init_state
        self.cur_state = init_state
        MDP.__init__(self, self.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

        # The following lines are used for efficiency
        legal_states = [(x, y) for room in rooms for x, y in room.points_in_room]
        legal_states.extend([(door.x, door.y) for door in doors])
        self.legal_states = set(legal_states)
        self.door_locs = set([(door.x, door.y) for door in doors])
        self.width = max(self.legal_states, key=lambda tup: tup[0])[0] + 1
        self.height = max(self.legal_states, key=lambda tup: tup[1])[1] + 1
        # TODO CREATE A DICTIONARY FROM ROOMS TO LEGAL STATES IN ROOMS WITHOUT DOORS

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        raise TypeError("(simple_rl): Reproduction of results not implemented for CleanUpMDP.")

    def _transition_func(self, state, action):
        '''
        :param state: The state
        :param action: The action
        :return: The next state you get if in state and perform action
        '''
        dx, dy = self.transition(action)
        new_x = state.x + dx
        new_y = state.y + dy
        copy = state.copy()
        if (new_x, new_y) not in self.legal_states:
            return copy

        visited_set = set()
        blocks, i = self._account_for_blocks(new_x, new_y, state, action, visited_set)
        if blocks is None:
            return copy
        while i >= 0:
            # This accounts for multiple blocks
            visited_set.add(i)
            if blocks[i].x == copy.blocks[i].x and blocks[i].y == copy.blocks[i].y:
                return copy
            copy.blocks = blocks
            blocks, i = self._account_for_blocks(blocks[i].x, blocks[i].y, copy, action, visited_set)
            if blocks is None:
                return state.copy()

        # if blocks is not None:
        if i >= 0 and blocks[i].x == copy.blocks[i].x and blocks[i].y == copy.blocks[i].y:
            return copy

        old_room = self.check_in_room(state.rooms, state.x, state.y)
        new_room = self.check_in_room(state.rooms, new_x, new_y)
        back_x = state.x - dx
        back_y = state.y - dy
        if (state.x, state.y) in self.door_locs or (new_x, new_y) in self.door_locs or \
                (back_x, back_y) in self.door_locs or old_room.__eq__(new_room):
            copy.blocks = blocks
            copy.x = new_x
            copy.y = new_y

        return copy

    @staticmethod
    def find_block(blocks, x, y):
        '''
        :param blocks: The list of blocks
        :param x: x coordinate in question
        :param y: y coordinate in question
        :return: The block (x, y) is associated with.  Or False if no association found.
        '''
        for block in blocks:
            if x == block.x and y == block.y:
                return block
        return False

    def _account_for_blocks(self, x, y, state, action, visited_set):
        '''
        :param x: X coordinate of the agent or block that just moved to that location
        :param y: Y coordinate of the agent or block that just moved to that location
        :param state: The current state
        :param action: The current action
        :param visited_set: The set of indices blocks that have already been visited
        :return:
        '''
        copy_blocks = state.blocks[:]
        if x == state.x and y == state.y:
            return copy_blocks, -1
        for i in range(len(state.blocks)):
            block = state.blocks[i]
            if i not in visited_set and x == block.x and y == block.y:
                dx, dy = self.transition(action)
                new_x = block.x + dx
                new_y = block.y + dy
                if (new_x, new_y) not in self.legal_states:
                    return None, -1
                else:
                    back_x = block.x - dx
                    back_y = block.y - dy
                    if (block.x, block.y) in self.door_locs or (new_x, new_y) in self.door_locs or \
                            (back_x, back_y) in self.door_locs:
                        block = block.copy()
                        block.x = new_x
                        block.y = new_y
                        copy_blocks[i] = block
                    else:
                        old_room = self.check_in_room(state.rooms, block.x, block.y)
                        new_room = self.check_in_room(state.rooms, new_x, new_y)
                        if old_room.__eq__(new_room):
                            block = block.copy()
                            block.x = new_x
                            block.y = new_y
                            copy_blocks[i] = block
                        else:
                            return None, -1
                    return copy_blocks, i
        return copy_blocks, -1

    @staticmethod
    def check_in_room(rooms, x, y):
        '''
        :param rooms: A list of rooms
        :param x: x coordinate
        :param y: y coordinate
        :return: Checks which room (x, y) is in.  Returns the room if the room is found.
                 Returns False otherwise.
        '''
        for room in rooms:
            if (x, y) in room.points_in_room:
                return room
        return False

    @staticmethod
    def transition(action):
        '''
        :param action: The action
        :return: A tuple for the delta x and y direction associated with that action
        '''
        dx = 0
        dy = 0
        if action == "up":
            dx = 0
            dy = 1
        elif action == "down":
            dx = 0
            dy = -1
        elif action == "left":
            dx = -1
            dy = 0
        elif action == "right":
            dx = 1
            dy = 0
        return dx, dy

    def _reward_func(self, state, action, next_state):
        '''
        :param state: The state you are in before performing the action
        :param action: The action you would like to perform in the state
        :param next_state: next state.
        :return: A double indicating how much reward to assign to that state.
                 1000.0 for the terminal state.  -1.0 for every other state.
        '''
        # next_state = self.transition_func(state, action)
        if self.is_terminal(self.task, state):
            return 0.0
        return 1000.0 if self.is_terminal(self.task, next_state) else -1.0

    @staticmethod
    def is_terminal(task, next_state):
        '''
        :param task: A CleanUpTask class
        :param next_state: The state we want to check is terminal
        :return: A boolean indicating whether the state is terminal or not
        '''
        if task.block_name is None:
            task_block = [block for block in next_state.blocks if block.color == task.block_color][0]
        else:
            task_block = [block for block in next_state.blocks if block.name == task.block_name][0]

        if task.goal_room_name is None:
            task_room = [room for room in next_state.rooms if room.color == task.goal_room_color][0]
        else:
            task_room = [room for room in next_state.rooms if room.name == task.goal_room_name][0]

        return task_room.contains(task_block)

    def __str__(self):
        # TODO WRITE OUT LATER
        return "CleanUpMDP: " + str(self.task)

    def reset(self):
        self.cur_state = copy.deepcopy(self.init_state)
        # if self.rand_init:
        #     block_loc = [(x, y) for block in blocks for (x, y) in (block.x, block.y)]
        #     new_loc = random.choice([(x, y) for room in self.init_state.rooms for (x, y) in
        #                              room.points_in_room if (x, y) not in block_loc])
        #     self.cur_state.x, self.cur_state.y = new_loc

    def visualize_agent(self, agent):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.cleanup.cleanup_visualizer import draw_state
        mdpv.visualize_agent(self, agent, draw_state)
        input("Press anything to quit ")

    def visualize_value(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.cleanup.cleanup_visualizer import draw_state
        mdpv.visualize_value(self, draw_state)
        input("Press anything to quit ")

    def visualize_interaction(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.cleanup.cleanup_visualizer import draw_state
        mdpv.visualize_interaction(self, draw_state)
        input("Press anything to quit ")

    def visualize_policy(self, policy):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.cleanup.cleanup_visualizer import draw_state
        mdpv.visualize_policy(self, policy=policy, draw_state=draw_state, action_char_dict={})
        input("Press anything to quit ")


if __name__ == "__main__":
    from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
    from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
    from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom

    task = CleanUpTask("green", "red")
    room1 = CleanUpRoom("room1", [(x, y) for x in range(5) for y in range(3)], "blue")
    block1 = CleanUpBlock("block1", 1, 1, color="green")
    block2 = CleanUpBlock("block2", 2, 4, color="purple")
    block3 = CleanUpBlock("block3", 8, 1, color="orange")
    room2 = CleanUpRoom("room2", [(x, y) for x in range(5, 10) for y in range(3)], color="red")
    room3 = CleanUpRoom("room3", [(x, y) for x in range(0, 10) for y in range(3, 6)], color="yellow")
    rooms = [room1, room2, room3]
    blocks = [block1, block2, block3]
    doors = [CleanUpDoor(4, 0), CleanUpDoor(3, 2)]
    mdp = CleanUpMDP(task, rooms=rooms, doors=doors, blocks=blocks)
    mdp.visualize_interaction()