''' Helper functions for executing actions in the Taxi Problem '''

# Other imports.
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject

def _is_wall_in_the_way(state, dx=0, dy=0):
    '''
    Args:
        state (TaxiState)
        dx (int) [optional]
        dy (int) [optional]

    Returns:
        (bool): true iff the new loc of the agent is occupied by a wall.
    '''
    for wall in state.objects["wall"]:
        if wall["x"] == state.objects["agent"][0]["x"] + dx and \
            wall["y"] == state.objects["agent"][0]["y"] + dy:
            return True
    return False

def _move_pass_in_taxi(state, dx=0, dy=0):
    '''
    Args:
        state (TaxiState)
        x (int) [optional]
        y (int) [optional]

    Returns:
        (list of dict): List of new passenger attributes.

    '''
    passenger_attr_dict_ls = state.get_objects_of_class("passenger")
    for i, passenger in enumerate(passenger_attr_dict_ls):
        if passenger["in_taxi"] == 1:
            passenger_attr_dict_ls[i]["x"] += dx
            passenger_attr_dict_ls[i]["y"] += dy

def is_taxi_terminal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers at at their destinations, not in the taxi.
    '''
    for p in state.get_objects_of_class("passenger"):
        if p.get_attribute("in_taxi") == 1 or p.get_attribute("x") != p.get_attribute("dest_x") or \
            p.get_attribute("y") != p.get_attribute("dest_y"):
            return False
    return True
