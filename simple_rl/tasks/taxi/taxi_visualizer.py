# Python imports.
from __future__ import print_function
try:
    import pygame
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")

# Other imports.
from simple_rl.utils.chart_utils import color_ls

def _draw_state(screen,
                taxi_oomdp,
                state,
                draw_statics=False,
                agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        taxi_oomdp (TaxiOOMDP)
        state (State)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''
    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / taxi_oomdp.width
    cell_height = (scr_height - height_buffer * 2) / taxi_oomdp.height
    objects = state.get_objects()
    agent_x, agent_y = objects["agent"][0]["x"], objects["agent"][0]["y"]

    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255,255,255), agent_shape)
        top_left_point = width_buffer + cell_width*(agent_x - 1), height_buffer + cell_height*(taxi_oomdp.height - agent_y)
        agent_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)

        # Draw new.
        agent_shape = _draw_agent(agent_center, screen, base_size=min(cell_width, cell_height)/2.5 - 4)
    else:
        top_left_point = width_buffer + cell_width*(agent_x - 1), height_buffer + cell_height*(taxi_oomdp.height - agent_y)
        agent_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
        agent_shape = _draw_agent(agent_center, screen, base_size=min(cell_width, cell_height)/2.5 - 4)

    # Do passengers first so the agent wipe out will wipe passengers, too.
    for i, p in enumerate(objects["passenger"]):
        # Passenger
        pass_x, pass_y = p["x"], p["y"]
        taxi_size = int(min(cell_width, cell_height) / 8.5) if p["in_taxi"] else int(min(cell_width, cell_height) / 5.0)
        top_left_point = int(width_buffer + cell_width*(pass_x - 1) + taxi_size + 38) , int(height_buffer + cell_height*(taxi_oomdp.height - pass_y) + taxi_size + 35)
        dest_col = (max(color_ls[-i-1][0]-30, 0), max(color_ls[-i-1][1]-30, 0), max(color_ls[-i-1][2]-30, 0))
        pygame.draw.circle(screen, dest_col, top_left_point, taxi_size)
    
    # Statics
    if draw_statics:
        # For each row:
        for i in range(taxi_oomdp.width):
            # For each column:
            for j in range(taxi_oomdp.height):
                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

        # Draw walls.
        for w in objects["wall"]:
            # Passenger
            w_x, w_y = w["x"], w["y"]
            top_left_point = width_buffer + cell_width*(w_x -1) + 5, height_buffer + cell_height*(taxi_oomdp.height - w_y) + 5
            pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width - 10, cell_height - 10), 0)

    for i, p in enumerate(objects["passenger"]):
        # Dest.
        dest_x, dest_y = p["dest_x"], p["dest_y"]
        top_left_point = int(width_buffer + cell_width*(dest_x - 1) + 25), int(height_buffer + cell_height*(taxi_oomdp.height - dest_y) + 25)
        dest_col = (int(max(color_ls[-i-1][0]-30, 0)), int(max(color_ls[-i-1][1]-30, 0)), int(max(color_ls[-i-1][2]-30, 0)))
        pygame.draw.rect(screen, dest_col, top_left_point + (cell_width / 6, cell_height / 6), 0)

    pygame.display.flip()

    return agent_shape

def _draw_agent(center_point, screen, base_size=30):
    '''
    Args:
        center_point (tuple): (x,y)
        screen (pygame.Surface)

    Returns:
        (pygame.rect)
    '''
    tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
    tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
    tri_top = center_point[0], center_point[1] - base_size
    tri = [tri_bot_left, tri_top, tri_bot_right]
    tri_color = (98, 140, 190)

    return pygame.draw.polygon(screen, tri_color, tri)
