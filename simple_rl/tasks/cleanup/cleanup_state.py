import copy
import random

from simple_rl.mdp.StateClass import State

from simple_rl.tasks.cleanup.CleanUpMDPClass import CleanUpMDP


class CleanUpState(State):
    def __init__(self, task, x, y, blocks=[], doors=[], rooms=[]):
        '''
        :param task: The given CleanUpTask
        :param x: Agent x coordinate
        :param y: Agent y coordinate
        :param blocks: List of blocks
        :param doors: List of doors
        :param rooms: List of rooms
        '''
        self.x = x
        self.y = y
        self.blocks = blocks
        self.doors = doors
        self.rooms = rooms
        self.task = task
        State.__init__(self, data=[task, (x, y), blocks, doors, rooms])

    def __hash__(self):
        alod = [tuple(self.data[i]) for i in range(1, len(self.data))]
        alod.append(self.data[0])
        return hash(tuple(alod))

    def __str__(self):
        str_builder = "(" + str(self.x) + ", " + str(self.y) + ")\n"
        str_builder += "\nBLOCKS:\n"
        for block in self.blocks:
            str_builder += str(block) + "\n"
        str_builder += "\nDOORS:\n"
        for door in self.doors:
            str_builder += str(door) + "\n"
        str_builder += "\nROOMS:\n"
        for room in self.rooms:
            str_builder += str(room) + "\n"
        return str_builder

    @staticmethod
    def list_eq(alod1, alod2):
        '''
        :param alod1: First list
        :param alod2: Second list
        :return: A boolean indicating whether or not the lists are the same
        '''
        if len(alod1) != len(alod2):
            return False
        sa = set(alod2)
        for item in alod1:
            if item not in sa:
                return False

        return True

    def __eq__(self, other):
        return isinstance(other, CleanUpState) and self.x == other.x and self.y == other.y and \
               self.list_eq(other.rooms, self.rooms) and self.list_eq(other.doors, self.doors) and \
               self.list_eq(other.blocks, self.blocks)

    def is_terminal(self):
        return CleanUpMDP.is_terminal(self.task, next_state=self)

    def copy(self):
        new_blocks = [block.copy() for block in self.blocks]
        new_rooms = [room.copy() for room in self.rooms]
        new_doors = [door.copy() for door in self.doors]
        return CleanUpState(self.task, self.x, self.y, new_blocks, new_doors, new_rooms)