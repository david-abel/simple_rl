'''
CleanupMDPClass.py: Contains the CleanupMDP class. 

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
import random
import copy

# simple_rl imports.
from ...mdp.oomdp.OOMDPClass import OOMDP
from ...mdp.oomdp.OOMDPObjectClass import OOMDPObject
from CleanupStateClass import CleanupState
import cleanup_action_helpers

class CleanupOOMDP(OOMDP):
    ''' Class for a Cleanup OO-MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]
    ATTRIBUTES = ["x", "y"]
    CLASSES = ["agent", "wall", "toy"]

    def __init__(self, width, height, agent_loc, walls, toys, slip_prob=0):
        init_state = self._create_init_state(height, width, agent_loc, walls, toys)
        OOMDP.__init__(self, CleanupOOMDP.ACTIONS, self.objects, self._cleanup_transition_func, self._cleanup_reward_func, init_state=init_state)
        self.height = height
        self.width = width
        self.slip_prob = slip_prob

    def _create_init_state(self, height, width, agent_loc, walls, toys):
        '''
        Args:
            height (int)
            width (int)
            agent_loc (dict): {key=attr_name : val=int}
            walls (list of dicts): [{key=attr_name : val=int, ... }, ...]
            toys (list of dicts): [{key=attr_name : val=int, ... }, ...]

        Returns:
            (OOMDP State)
        '''

        self.objects = {c : [] for c in CleanupOOMDP.CLASSES}

        # Make agent.
        agent_attributes = {}
        for attr in agent_loc.keys():
            agent_attributes[attr] = agent_loc[attr]
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
        for t in toys:
            toy_attributes = {}
            for attr in toy:
                toy_attributes[attr] = t[attr]
            toy = OOMDPObject(attributes=toy_attributes, name="toy")
            self.objects["toy"].append(toy)

        return CleanupState(self.objects)

    def _cleanup_reward_func(self, state, action):
        '''
        Args:
            state (OOMDP State)
            action (str)

        Returns
            (float)
        '''
        _error_check(state, action)

        next_state = self._cleanup_transition_func(state, action)

        if next_state.is_terminal():
            return 10
        else:
            return -0.01

    def _cleanup_transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        _error_check(state, action)

        if self.slip_prob > random.random():
            # Flip dir.
            if action == "up":
                action = "down"
            elif action == "down":
                action = "up"
            elif action == "left":
                action = "right"
            elif action == "right":
                action = "left"

        if action == "up" and state.get_agent_y() < self.height:
            state = cleanup_action_helpers.move_agent(state, self.slip_prob, dy=1)
        elif action == "down" and state.get_agent_y() > 1:
            state = cleanup_action_helpers.move_agent(state, self.slip_prob, dy=-1)
        elif action == "right" and state.get_agent_x() < self.width:
            state = cleanup_action_helpers.move_agent(state, self.slip_prob, dx=1)
        elif action == "left" and state.get_agent_x() > 1:
            state = cleanup_action_helpers.move_agent(state, self.slip_prob, dx=-1)
        elif action == "dropoff":
            state = cleanup_action_helpers.agent_dropoff(state)
        elif action == "pickup":
            state = cleanup_action_helpers.agent_pickup(state)
        
        # Make terminal.
        if is_cleanup_terminal_state(state):
            state.set_terminal(True)
        
        # All OOMDP states must be updated.
        state._update()
        
        return state

    def __str__(self):
        return "cleanup_h-" + str(self.height) + "_w-" + str(self.width)

    def _is_toy_in_room(self, state, toy, room):
        '''
        Args:
            toy (OOMDPObject)
            room (tuple): < (x_1, y_1), (x_2, y_2) >

        Returns:
            True iff toy \in @room
        '''
        rx_1, ry_1 = room[0][0], room[0][1]
        rx_2, ry_2 = room[1][0], room[1][1]

        for t in state.objects["toy"]:
            if t == toy:
                if (rx_1 <= t["x"] <= rx_2 or rx_2 <= t["x"] <= rx_1) and \
                    (ry_1 <= t["y"] <= ry_2 or ry_2 <= t["y"] <= ry_1):
                    return True
        return False


def is_cleanup_terminal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers at at their destinations, not in the Cleanup.
    '''
    for p in state.objects["passenger"]:
        if p.get_attribute("in_cleanup") == 1 or p.get_attribute("x") != p.get_attribute("dest_x") or \
            p.get_attribute("y") != p.get_attribute("dest_y"):
            return False
    return True

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in CleanupOOMDP.ACTIONS:
        print "Error: the action provided (" + str(action) + ") was invalid."
        quit()

    if not isinstance(state, CleanupState):
        print "Error: the given state (" + str(state) + ") was not of the correct class."
        quit()

def main():
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":8, "y":4, "dest_x":2, "dest_y":2, "in_Cleanup":0}]
    cleanup_world = CleanupOOMDP(10, 10, agent_loc=agent, walls=[], passengers=passengers)

if __name__ == "__main__":
    main()
