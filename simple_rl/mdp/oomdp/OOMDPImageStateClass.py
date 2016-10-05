'''
OOMDPImageStateClass.py: Contains implementation for creating an
    Object Oriented MDP State from an Image.
'''

# Python imports.
import itertools
import numpy

# Local imports.
from OOMDPStateClass import OOMDPState
from OOMDPObjectClass import OOMDPObject
from ..ImageStateClass import ImageState

class OOMDPImageState(OOMDPState, ImageState):
    ''' OOMDP State from an Image class '''

    def __init__(self, image, other_data=None):
        '''
        Args:
            objects (dict of OOMDPObject instances): {key=object class (str):val = object instances}
        '''
        objects = self._objects_from_image(image)
        
        ImageState.__init__(self, image=image)
        OOMDPState.__init__(self, objects=objects)

    def _objects_from_image(self, image):
        '''
        Args:
            image (numpy array)

        Returns:
            (list of OOMDPObjects)
        '''
        center_points = self._object_center_points_from_image(image)
        objects = {"all":[]}
        for point in center_points:
            attributes = {"x":point[0], "y":point[1]}
            new_object = OOMDPObject(attributes, name="OOMDPObject(x:" + str(point[0]) + ",y:" + str(point[1]) + ")")
            objects["all"] += [new_object]
        return objects

    def _object_center_points_from_image(self, image):
        '''
        Args:
            image (numpy array)

        Returns:
            (list of [x,y])

        Notes:
            Code taken from Stack Overflow user Jaime's answer, here:
                http://stackoverflow.com/questions/14538168/finding-bounding-boxes-of-rgb-colors-in-image
        '''

        r, g, b = [numpy.unique(image[..., j]) for j in (0, 1, 2)]
        combos = itertools.product(r, g, b)
        center_points = []
        for r0, g0, b0 in combos:
            rows, cols = numpy.where((image[..., 0] == r0) &
                                    (image[..., 1] == g0) &
                                    (image[..., 2] == b0))
            if len(rows):
                center_points.append([(numpy.min(rows) + numpy.max(rows)) / 2,\
                                    (numpy.min(cols) + numpy.max(cols)) / 2])
                # bounding_boxes[(r0, g0, b0)] = (numpy.min(rows), numpy.max(rows),
                                                # numpy.min(cols), numpy.max(cols))
        return center_points

def _is_similar_color(color_one, color_two, threshold = 20):
    return sum([abs(color_one[i] - color_two[i]) for i in xrange(len(color_one))]) <= threshold

if __name__ == "__main__":
    main()
