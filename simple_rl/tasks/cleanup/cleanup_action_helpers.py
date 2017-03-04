''' Helper functions for executing actions in the Taxi Problem '''

import random
import copy

def move_agent(state, dx=0, dy=0):
    '''
    Args:
        state (TaxiState)
        dx (int)
        dy (int)

    Returns:
        (TaxiState)
    '''
    if _is_wall_in_the_way(state, dx=dx, dy=dy):
        # There's a wall in the way.
        return state

    _move_toys(state, dx, dy)

    new_state.objects["agent"][0]["x"] += dx
    new_state.objects["agent"][0]["y"] += dy

    return new_state

def _is_wall_in_the_way(state, dx, dy):
    '''
    Args:
        state (TaxiState)
        dx (int)
        dy (int)

    Returns:
        (bool): true iff the new loc of the agent is occupied by a wall.
    '''
    new_ax, new_ay = state.objects["agent"][0]["x"] + dx, state.objects["agent"][0]["y"] + dy
    for wall in state.objects["wall"]:
        if wall["x"] == new_ax and wall["y"] == new_ay:
            return True

    return False

def _move_toys(state, dx, dy):
    new_ax, new_ay = state.objects["agent"][0]["x"] + dx, state.objects["agent"][0]["y"] + dy

    for toy in state.objects["toys"]:
        if toy["x"] == new_ax and toy["y"] == new_ay:
            



