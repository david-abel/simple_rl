class CleanUpBlock:

    def __init__(self, name, x=0, y=0, color=""):
        self.name = name
        self.x = x
        self.y = y
        self.color = color

    @staticmethod
    def class_name():
        return "block"

    def name(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, CleanUpBlock) and self.x == other.x and self.y == other.y and self.name == other.name \
               and self.color == other.color

    def __hash__(self):
        return hash(tuple([self.name, self.x, self.y, self.color]))

    def copy_with_name(self, new_name):
        return CleanUpBlock(new_name, x=self.x, y=self.y, color=self.color)

    def copy(self):
        return CleanUpBlock(name=self.name, x=self.x, y=self.y, color=self.color)

    def __str__(self):
        return "BLOCK.  Name: " + self.name + ", (x,y): (" + str(self.x) + "," + str(self.y) + "), Color: " + self.color
