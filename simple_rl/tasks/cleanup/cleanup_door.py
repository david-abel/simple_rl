class CleanUpDoor:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash(tuple([self.x, self.y]))

    def copy(self):
        return CleanUpDoor(self.x, self.y)

    def __eq__(self, other):
        return isinstance(other, CleanUpDoor) and self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))