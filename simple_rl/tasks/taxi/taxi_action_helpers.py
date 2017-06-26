''' Helper functions for executing actions in the Taxi Problem '''

import random
import copy

def move_agent(state, slip_prob=0, dx=0, dy=0):
    '''
    Args:
        state (TaxiState)
        dx (int) [optional]
        dy (int) [optional]

    Returns:
        (TaxiState)
    '''
    if _is_wall_in_the_way(state, dx=dx, dy=dy):
        # There's a wall in the way.
        return state

    new_state = _move_passenger_in_taxi(state, x=dx, y=dy)
    new_state.objects["agent"][0]["x"] += dx
    new_state.objects["agent"][0]["y"] += dy

    return new_state

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

def _move_passenger_in_taxi(state, x=0, y=0):
    '''
    Args:
        state (TaxiState)
        x (int) [optional]
        y (int) [optional]

    Returns
        (TaxiState)
    '''
    for i, passenger in enumerate(state.objects["passenger"]):
        if passenger["in_taxi"] == 1:
            state.objects["passenger"][i]["x"] += x
            state.objects["passenger"][i]["y"] += y
    
    return state

def agent_pickup(state):
    '''
    Args:
        state (TaxiState)

    Returns
        (TaxiState)
    '''
    if state.objects["agent"][0]["has_passenger"] == 0:
        # If the agent doesn't already have a passenger.
        
        for i, passenger in enumerate(state.objects["passenger"]):
            if state.objects["agent"][0]["x"] == passenger["x"] and \
                state.objects["agent"][0]["y"] == passenger["y"]:

                # If they're at the same location, pickup.
                state.objects["agent"][0]["has_passenger"] = 1
                state.objects["passenger"][i]["in_taxi"] = 1
    return state

def agent_dropoff(state):
    '''
    Args:
        state (TaxiState)

    Returns
        (TaxiState)
    '''
    if state.objects["agent"][0]["has_passenger"] == 1:
        # If the agent has a passenger.
        for i, passenger in enumerate(state.objects["passenger"]):

            if passenger["in_taxi"] == 1:
                # Drop off the passenger.
                state.objects["passenger"][i]["in_taxi"] = 0
                state.objects["agent"][0]["has_passenger"] = 0
    return state
