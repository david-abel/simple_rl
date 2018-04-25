'''
TaxiMDPClass.py: Contains the TaxiMDP class.

From:
    Dietterich, Thomas G. "Hierarchical reinforcement learning with the
    MAXQ value function decomposition." J. Artif. Intell. Res.(JAIR) 13
    (2000): 227-303.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
from __future__ import print_function
import random
import copy

# Other imports.
from simple_rl.mdp.oomdp.OOMDPClass import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.tasks.taxi.TaxiStateClass import TaxiState
from simple_rl.tasks.taxi import taxi_helpers


class TaxiOOMDP(OOMDP):
    ''' Class for a Taxi OO-MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff"]
    ATTRIBUTES = ["x", "y", "has_passenger", "in_taxi", "dest_x", "dest_y"]
    CLASSES = ["agent", "wall", "passenger"]

    def __init__(self, width, height, agent, walls, passengers, slip_prob=0, gamma=0.99):
        self.height = height
        self.width = width

        agent_obj = OOMDPObject(attributes=agent, name="agent")
        wall_objs = self._make_oomdp_objs_from_list_of_dict(walls, "wall")
        pass_objs = self._make_oomdp_objs_from_list_of_dict(passengers, "passenger")

        init_state = self._create_state(agent_obj, wall_objs, pass_objs)
        OOMDP.__init__(self, TaxiOOMDP.ACTIONS, self._taxi_transition_func, self._taxi_reward_func, init_state=init_state, gamma=gamma)
        self.slip_prob = slip_prob

    def _create_state(self, agent_oo_obj, walls, passengers):
        '''
        Args:
            agent_oo_obj (OOMDPObjects)
            walls (list of OOMDPObject)
            passengers (list of OOMDPObject)

        Returns:
            (OOMDP State)

        TODO: Make this more egneral and put it in OOMDPClass.
        '''

        objects = {c : [] for c in TaxiOOMDP.CLASSES}

        objects["agent"].append(agent_oo_obj)

        # Make walls.
        for w in walls:
            objects["wall"].append(w)

        # Make passengers.
        for p in passengers:
            objects["passenger"].append(p)

        return TaxiState(objects)

    def _taxi_reward_func(self, state, action):
        '''
        Args:
            state (OOMDP State)
            action (str)

        Returns
            (float)
        '''
        _error_check(state, action)

        # Stacked if statements for efficiency.
        if action == "dropoff":
            # If agent is dropping off.
            agent = state.get_first_obj_of_class("agent")

            # Check to see if all passengers at destination.
            if agent.get_attribute("has_passenger"):
                for p in state.get_objects_of_class("passenger"):
                    if p.get_attribute("x") != p.get_attribute("dest_x") or p.get_attribute("y") != p.get_attribute("dest_y"):
                        return 0 - self.step_cost
                return 1 - self.step_cost
        return 0 - self.step_cost

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

        if action == "up" and state.get_agent_y() < self.height:
            next_state = self.move_agent(state, self.slip_prob, dy=1)
        elif action == "down" and state.get_agent_y() > 1:
            next_state = self.move_agent(state, self.slip_prob, dy=-1)
        elif action == "right" and state.get_agent_x() < self.width:
            next_state = self.move_agent(state, self.slip_prob, dx=1)
        elif action == "left" and state.get_agent_x() > 1:
            next_state = self.move_agent(state, self.slip_prob, dx=-1)
        elif action == "dropoff":
            next_state = self.agent_dropoff(state)
        elif action == "pickup":
            next_state = self.agent_pickup(state)
        else:
            next_state = state

        # Make terminal.
        if taxi_helpers.is_taxi_terminal_state(next_state):
            next_state.set_terminal(True)

        # All OOMDP states must be updated.
        next_state.update()

        return next_state

    def __str__(self):
        return "taxi_h-" + str(self.height) + "_w-" + str(self.width)

    def visualize_agent(self, agent):
        from ...utils.mdp_visualizer import visualize_agent
        from taxi_visualizer import _draw_state
        visualize_agent(self, agent, _draw_state)
        _ = input("Press anything to quit ")
        sys.exit(1)

    def visualize_interaction(self):
        from simple_rl.utils.mdp_visualizer import visualize_interaction
        from taxi_visualizer import _draw_state
        visualize_interaction(self, _draw_state)
        raw_input("Press anything to quit ")
        sys.exit(1)


    # ----------------------------
    # -- Action Implementations --
    # ----------------------------

    def move_agent(self, state, slip_prob=0, dx=0, dy=0):
        '''
        Args:
            state (TaxiState)
            dx (int) [optional]
            dy (int) [optional]

        Returns:
            (TaxiState)
        '''

        if taxi_helpers._is_wall_in_the_way(state, dx=dx, dy=dy):
            # There's a wall in the way.
            return state

        next_state = copy.deepcopy(state)

        # Move Agent.
        agent_att = next_state.get_first_obj_of_class("agent").get_attributes()
        agent_att["x"] += dx
        agent_att["y"] += dy

        # Move passenger.
        taxi_helpers._move_pass_in_taxi(next_state, dx=dx, dy=dy)

        return next_state

    def agent_pickup(self, state):
        '''
        Args:
            state (TaxiState)

        '''
        next_state = copy.deepcopy(state)

        agent = next_state.get_first_obj_of_class("agent")

        # update = False
        if agent.get_attribute("has_passenger") == 0:

            # If the agent does not have a passenger.
            for i, passenger in enumerate(next_state.get_objects_of_class("passenger")):
                if agent.get_attribute("x") == passenger.get_attribute("x") and agent.get_attribute("y") == passenger.get_attribute("y"):
                    # Pick up passenger at agent location.
                    agent.set_attribute("has_passenger", 1)
                    passenger.set_attribute("in_taxi", 1)

        return next_state

    def agent_dropoff(self, state):
        '''
        Args:
            state (TaxiState)

        Returns:
            (TaxiState)
        '''
        next_state = copy.deepcopy(state)

        # Get Agent, Walls, Passengers.
        agent = next_state.get_first_obj_of_class("agent")
        # agent = OOMDPObject(attributes=agent_att, name="agent")
        passengers = next_state.get_objects_of_class("passenger")

        if agent.get_attribute("has_passenger") == 1:
            # Update if the agent has a passenger.
            for i, passenger in enumerate(passengers):

                if passenger.get_attribute("in_taxi") == 1:
                    # Drop off the passenger.
                    passengers[i].set_attribute("in_taxi", 0)
                    agent.set_attribute("has_passenger", 0)

        return next_state

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in TaxiOOMDP.ACTIONS:
        raise ValueError("Error: the action provided (" + str(action) + ") was invalid.")

    if not isinstance(state, TaxiState):
        raise ValueError("Error: the given state (" + str(state) + ") was not of the correct class.")


def main():
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":8, "y":4, "dest_x":2, "dest_y":2, "in_taxi":0}]
    taxi_world = TaxiOOMDP(10, 10, agent=agent, walls=[], passengers=passengers)

if __name__ == "__main__":
    main()
