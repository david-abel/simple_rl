from simple_rl.tasks.cleanup.cleanup_state import CleanUpState

class CleanUpRoom:
    def __init__(self, name, points_in_room=[(x + 1, y + 1) for x in range(24) for y in range(24)], color="blue"):
        self.name = name
        self.points_in_room = points_in_room
        self.color = color

    def contains(self, block):
        return (block.x, block.y) in self.points_in_room

    def copy(self):
        return CleanUpRoom(self.name, self.points_in_room[:], color=self.color)

    def __hash__(self):
        return hash(tuple([self.name, self.color, tuple(self.points_in_room)]))

    def __eq__(self, other):
        if not isinstance(other, CleanUpRoom):
            return False

        return self.name == other.name and self.color == other.color and \
               CleanUpState.list_eq(self.points_in_room, other.points_in_room)

    def __str__(self):
        return "color: " + self.color + ", points: " + " ".join(
            str(tup) for tup in self.points_in_room)
