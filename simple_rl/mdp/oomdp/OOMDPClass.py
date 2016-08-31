'''
OOMDPClass.py: Implementation for Object-Oriented MDPs.

From:
	Diuk, Carlos, Andre Cohen, and Michael L. Littman.
	"An object-oriented representation for efficient reinforcement learning."
	Proceedings of the 25th international conference on Machine learning. ACM, 2008.
'''

# Local libs.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.chain.ChainStateClass import ChainState

class OOMDP(MDP):
    ''' Abstract class for an Object Oriented Markov Decision Process. '''
    
    def __init__(self, actions, objects, transition_func, reward_func, init_state, gamma=0.95):
        MDP.__init__(self, actions, transition_func, reward_func, init_state=init_state, gamma=gamma)
        self.objects = objects
