'''
RBFCodingClass.py: Class for radial basis function feature coding.

Author: David Abel
'''

# Python imports.
import math

class RBFCoding(object):

    NAME = "rbf_coding"

    def get_features(self, state):
        '''
        Args:
            state (simple_rl.State)

        Returns:
            (list): contains the radial basis function features.
        '''
        state_feats = state.features()

        # Perform feature bucketing.        
        new_features = []
        for i, feat in enumerate(state_feats):
            new_feat = math.exp(-(feat)**2)
            new_features.append(new_feat)

        return new_features
