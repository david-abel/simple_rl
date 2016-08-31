'''
TaxiMDPClass.py: Contains the TaxiMDP class. 

From:
    Dietterich, Thomas G. "Hierarchical reinforcement learning with the
    MAXQ value function decomposition." J. Artif. Intell. Res.(JAIR) 13
    (2000): 227-303.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# simple_rl imports.
from simple_rl.mdp.oomdp.OOMDPClass import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.tasks.taxi.TaxiStateClass import TaxiState
from simple_rl.tasks.taxi import taxi_action_helpers

class TaxiOOMDP(OOMDP):
    ''' Class for a Taxi OO-MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff"]
    ATTRIBUTES = ["x", "y", "has_passenger", "in_taxi", "dest_x", "dest_y"]
    CLASSES = ["agent", "wall", "passenger"]

    def __init__(self, height, width, agent_loc, walls, passengers):
        init_state = self._create_init_state(height, width, agent_loc, walls, passengers)
        OOMDP.__init__(self, TaxiOOMDP.ACTIONS, self.objects, self._transition_func, self._reward_func, init_state=init_state)
        self.height = height
        self.width = width

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

        self.objects = {attr : [] for attr in TaxiOOMDP.CLASSES}

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


    def _reward_func(self, state, action):
        '''
        Args:
            state (OOMDP State)
            action (str)

        Returns
            (float)
        '''
        _error_check(state, action)

        if is_taxi_terminal_state(state):
            return 1
        else:
            return 0

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        _error_check(state, action)

        if action == "up" and state.get_agent_y() < self.height:
            state = taxi_action_helpers.move_agent(state, dy=1)
        elif action == "down" and state.get_agent_y() > 1:
            state = taxi_action_helpers.move_agent(state, dy=-1)
        elif action == "right" and state.get_agent_x() < self.width:
            state = taxi_action_helpers.move_agent(state, dx=1)
        elif action == "left" and state.get_agent_x() > 1:
            state = taxi_action_helpers.move_agent(state, dx=-1)
        elif action == "dropoff":
            state = taxi_action_helpers.agent_dropoff(state)
        elif action == "pickup":
            state = taxi_action_helpers.agent_pickup(state)
        return state

    def __str__(self):
        return "taxi_h-" + str(self.height) + "_w-" + str(self.width)

def is_taxi_terminal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers at at their destinations.
    '''
    for p in state.objects["passenger"]:
        if p.get_attribute("x") != p.get_attribute("dest_x") or \
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
