'''
BucketCodingClass.py: Class for a simple bucketing method for feature values.

Summary: Performs a simple discretization on a given continuous state.

Author: David Abel
'''

# Python imports.
import math

class BucketCoding(object):

    NAME = "bucket_coding"

    def __init__(self, feature_max_vals, num_buckets):
        '''
        Args:
            feature_max_vals (list): The i-th element is an int/float indicating the max
                value that the i-th feature value can take on.
            num_buckets (int): Buckets each feature into this many buckets.
        '''
        self.feature_max_vals = feature_max_vals
        self.num_buckets = num_buckets

    def get_features(self, state):
        '''
        Args:
            state (simple_rl.State)

        Returns:
            (list): contains the bucketed features.
        '''
        state_feats = state.features()

        # Perform feature bucketing.        
        bucketed_features = []
        for i, feat in enumerate(state_feats):
            bucket_num = int(math.floor(self.num_buckets * feat / self.feature_max_vals[i]))
            bucketed_features.append(bucket_num)

        return bucketed_features
