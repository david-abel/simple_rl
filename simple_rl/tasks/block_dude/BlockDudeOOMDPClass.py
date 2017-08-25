'''
BlockDudeMDPClass.py: Contains the BlockDudeMDP class from the TI-89 Calculator.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
import random
import copy

# Other imports.
from simple_rl.mdp.oomdp.OOMDPClass import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.tasks.block_dude.BlockDudeStateClass import BlockDudeState
import block_dude_helpers as bd_helpers

class BlockDudeOOMDP(OOMDP):
    ''' Class for a BlockDude OO-MDP '''

    # Static constants.
    ACTIONS = ["climb", "left", "right", "pickup", "drop"]
    ATTRIBUTES = ["x", "y", "face_left", "face_right", "carried", "carrying"]
    CLASSES = ["wall", "block", "agent", "exit"]

    def __init__(self, width, height, agent, walls, blocks, exit, gamma=0.99):
        init_state = self._create_init_state(height, width, agent, walls, blocks, exit)
        OOMDP.__init__(self, BlockDudeOOMDP.ACTIONS, self.objects, self._block_dude_transition_func, self._block_dude_reward_func, init_state=init_state, gamma=gamma)
        self.height = height
        self.width = width

    def _create_init_state(self, height, width, agent, walls, blocks, exit):
        '''
        Args:
            height (int)
            width (int)
            agent (dict)
            walls (list of dicts)
            blocks (list of dicts)
            exit (dict)

        Returns:
            (OOMDP State)
        '''

        self.objects = {c : [] for c in BlockDudeOOMDP.CLASSES}

        # Make agent.
        agent_attributes = {}
        for attr in agent.keys():
            agent_attributes[attr] = agent[attr]
        agent = OOMDPObject(attributes=agent_attributes, name="agent")
        self.objects["agent"].append(agent)

        # Make walls.
        for w in walls:
            wall_attributes = {}
            for attr in w:
                wall_attributes[attr] = w[attr]
            wall = OOMDPObject(attributes=wall_attributes, name="wall")
            self.objects["wall"].append(wall)

        # Make passengers.
        for b in blocks:
            passenger_attributes = {}
            for attr in p:
                passenger_attributes[attr] = p[attr]
            block = OOMDPObject(attributes=passenger_attributes, name="block")
            self.objects["block"].append(block)

        # Make exit.
        exit_attributes = {}
        for attr in exit.keys():
            exit_attributes[attr] = exit[attr]
        exit = OOMDPObject(attributes=exit_attributes, name="exit")
        self.objects["exit"].append(exit)

        return BlockDudeState(self.objects)

    def _block_dude_reward_func(self, state, action):
        '''
        Args:
            state (OOMDP State)
            action (str)

        Returns
            (float)
        '''
        _error_check(state, action)

        next_state = self._block_dude_transition_func(state, action)
        
        if state.is_terminal():
            return 0
        if next_state.is_terminal():
            return 1
        else:
            return 0

    def _block_dude_transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        _error_check(state, action)

        next_state = copy.deepcopy(state)

        if action == "climb" and state.get_agent_y() < self.height:
            bd_helpers.climb(next_state)
        elif action == "right" and state.get_agent_x() < self.width:
            bd_helpers.move_agent(next_state, dx=1)
        elif action == "left" and state.get_agent_x() > 1:
            bd_helpers.move_agent(next_state, dx=-1)
        elif action == "drop":
            bd_helpers.drop(next_state)
        elif action == "pickup":
            bd_helpers.pickup(next_state)
        
        # Make terminal.
        if is_block_dude_terminal_state(next_state):
            next_state.set_terminal(True)
        
        # All OOMDP states must be updated.
        next_state.update()
        
        return next_state

    def __str__(self):
        return "blockdude_h-" + str(self.height) + "_w-" + str(self.width)

def is_block_dude_terminal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers at at their destinations, not in the BlockDude.
    '''
    ax, ay = state.get_agent_x(), state.get_agent_y()
    exit = state.get_first_obj_of_class("exit")
    goal_x, goal_y = exit["x"], exit["y"]

    return ax == goal_x and ay == goal_y

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in BlockDudeOOMDP.ACTIONS:
        print "Error: the action provided (" + str(action) + ") was invalid."
        quit()

    if not isinstance(state, BlockDudeState):
        print "Error: the given state (" + str(state) + ") was not of the correct class."
        quit()

def main():
    pass

if __name__ == "__main__":
    main()
