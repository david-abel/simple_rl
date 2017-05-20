'''
TaxiMDPClass.py: Contains the TaxiMDP class. 

From:
    Dietterich, Thomas G. "Hierarchical reinforcement learning with the
    MAXQ value function decomposition." J. Artif. Intell. Res.(JAIR) 13
    (2000): 227-303.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
import random
import copy

# simple_rl imports.
from ...mdp.oomdp.OOMDPClass import OOMDP
from ...mdp.oomdp.OOMDPObjectClass import OOMDPObject
from TaxiStateClass import TaxiState
import taxi_action_helpers

class TaxiOOMDP(OOMDP):
    ''' Class for a Taxi OO-MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff"]
    ATTRIBUTES = ["x", "y", "has_passenger", "in_taxi", "dest_x", "dest_y"]
    CLASSES = ["agent", "wall", "passenger"]

    def __init__(self, width, height, agent_loc, walls, passengers, slip_prob=0):
        init_state = self._create_init_state(height, width, agent_loc, walls, passengers)
        OOMDP.__init__(self, TaxiOOMDP.ACTIONS, self.objects, self._taxi_transition_func, self._taxi_reward_func, init_state=init_state)
        self.height = height
        self.width = width
        self.slip_prob = slip_prob

    def _create_init_state(self, height, width, agent_loc, walls, passengers):
        '''
        Args:
            height (int)
            width (int)
            agent_loc (dict): {key=attr_name : val=int}
            walls (list of dicts): [{key=attr_name : val=int, ... }, ...]
            passengers (list of dicts): [{key=attr_name : val=int, ... }, ...]

        Returns:
            (OOMDP State)
        '''

        self.objects = {c : [] for c in TaxiOOMDP.CLASSES}

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
        for p in passengers:
            passenger_attributes = {}
            for attr in p:
                passenger_attributes[attr] = p[attr]
            passenger = OOMDPObject(attributes=passenger_attributes, name="passenger")
            self.objects["passenger"].append(passenger)

        return TaxiState(self.objects)

    def _taxi_reward_func(self, state, action):
        '''
        Args:
            state (OOMDP State)
            action (str)

        Returns
            (float)
        '''
        _error_check(state, action)

        next_state = self._taxi_transition_func(state, action)

        if next_state.is_terminal():
            return 1
        else:
            return 0

    def _taxi_transition_func(self, state, action):
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

        next_state = copy.deepcopy(state)

        if action == "up" and state.get_agent_y() < self.height:
            next_state = taxi_action_helpers.move_agent(next_state, self.slip_prob, dy=1)
        elif action == "down" and state.get_agent_y() > 1:
            next_state = taxi_action_helpers.move_agent(next_state, self.slip_prob, dy=-1)
        elif action == "right" and state.get_agent_x() < self.width:
            next_state = taxi_action_helpers.move_agent(next_state, self.slip_prob, dx=1)
        elif action == "left" and state.get_agent_x() > 1:
            next_state = taxi_action_helpers.move_agent(next_state, self.slip_prob, dx=-1)
        elif action == "dropoff":
            next_state = taxi_action_helpers.agent_dropoff(next_state)
        elif action == "pickup":
            next_state = taxi_action_helpers.agent_pickup(next_state)
        else:
            next_state = next_state
        
        # Make terminal.
        if is_taxi_terminal_state(next_state):
            next_state.set_terminal(True)
        
        # All OOMDP states must be updated.
        next_state._update()
        
        return next_state

    def __str__(self):
        return "taxi_h-" + str(self.height) + "_w-" + str(self.width)

def is_taxi_terminal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers at at their destinations, not in the taxi.
    '''
    for p in state.objects["passenger"]:
        if p.get_attribute("in_taxi") == 1 or p.get_attribute("x") != p.get_attribute("dest_x") or \
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

    if action not in TaxiOOMDP.ACTIONS:
        print "Error: the action provided (" + str(action) + ") was invalid."
        quit()

    if not isinstance(state, TaxiState):
        print "Error: the given state (" + str(state) + ") was not of the correct class."
        quit()

def main():
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":8, "y":4, "dest_x":2, "dest_y":2, "in_taxi":0}]
    taxi_world = TaxiOOMDP(10, 10, agent_loc=agent, walls=[], passengers=passengers)

if __name__ == "__main__":
    main()
