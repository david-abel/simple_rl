''' additional_datastructures.py: File containing custom utility data structures for use in simple_rl. '''
import json

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


class TupleEncoder(json.JSONEncoder):
    '''
        A simple class for adding tuple encoding to json, from:
            https://stackoverflow.com/questions/15721363/preserve-python-tuples-with-json
    '''
    def encode(self, obj):
        '''
        Args:
            obj (Object): Arbitrary object to encode in JSON.

        Summary:
            Converts all tuples into dictionaries of two elements:
                (1) "tuple":true
                (2) "items:"<tuple_contents>
            To be used with the below static method (hinted_tuple_hook) to encode/decode json tuples.
        '''
        
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            else:
                return item

        return json.JSONEncoder.encode(self, hint_tuples(obj))

    @staticmethod
    def hinted_tuple_hook(obj):
        if '__tuple__' in obj.keys():
            return tuple(obj['items'])
        else:
            return obj
