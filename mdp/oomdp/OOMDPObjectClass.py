''' OOMDPObjectClass.py: Contains the OOMDP Object Class. '''

class OOMDPObject(object):
    ''' Abstract OOMDP Object class '''

    def __init__(self, attributes, name="OOMDP-Object"):
        '''
        Args:
            attributes (dict): {key=attr_name, val=int}
        '''
        self.name = name
        self.attributes = attributes

    def set_attribute(self, attr, val):
        self.attributes[attr] = val

    def get_attribute(self, attr):
        return self.attributes[attr]

    def get_obj_state(self):
        return self.attributes.values()

    def get_attributes(self):
        return self.attributes

    def __getitem__(self, key):
        return self.attributes[key]

    def __setitem__(self, key, item):
        self.attributes[key] = item

    def __str__(self):
        result = "o:" + self.name + " ["
        for attr in self.attributes:
            result += "a:" + str(attr) + " = " + str(self.attributes[attr]) + ", "
        return result + "]"
