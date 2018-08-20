''' additional_datastructures.py: File containing custom utility data structures for use in simple_rl. '''

class SimpleRLStack(object):
    ''' Implementation for a basic Stack data structure '''
    def __init__(self, _list=None):
        '''
        Args:
            _list (list) : underlying elements in the stack
        '''
        self._list = _list if _list is not None else []

    def __repr__(self):
        r = ''
        for element in self._list:
            r += str(element) + ', '
        return r

    def push(self, element):
        return self._list.append(element)

    def pop(self):
        if len(self._list) > 0:
            return self._list.pop()
        return None

    def peek(self):
        if len(self._list) > 0:
            return self._list[-1]
        return None

    def is_empty(self):
        return len(self._list) == 0

    def size(self):
        return len(self._list)
