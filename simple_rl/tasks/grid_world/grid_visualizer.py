try:
    import pygame
except ImportError:
    print "Warning: pygame not installed (needed for visuals)."
import random
import sys
from ...utils.ValueIterationClass import ValueIteration
from ..four_room.FourRoomMDPClass import FourRoomMDP

def _draw_state(screen, grid_mdp, state, state_text_dict={}, draw_statics=False, agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        grid_mdp (MDP)
        state (State)
        state_text_dict (dict):
            Key: state_text_dict
            Val: text (to write to the state)
        draw_grid
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_locs = grid_mdp.get_goal_locs()
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)

    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255,255,255), agent_shape)
        top_left_point = width_buffer + cell_width*(state.x - 1), height_buffer + cell_height*(grid_mdp.height - state.y)
        tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)

        # Draw new.
        agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 4)

    # Draw the static entities.
    if draw_statics:
        # For each row:
        for i in range(grid_mdp.width):
            # For each column:
            for j in range(grid_mdp.height):

                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                if state_text_dict != {}:
                    text_center_point = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    text = state_text_dict[i][j]
                    text_rendered = reg_font.render(text, True, (46, 49, 49))
                    screen.blit(text_rendered, text_center_point)

                if isinstance(grid_mdp, FourRoomMDP) and grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                    r = pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width-10, cell_height-10), 0)

                if (i+1,grid_mdp.height - j) in goal_locs:
                    # Draw goal.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (154, 195, 157)
                    pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 3.0))

                    # Goal text.                
                    text = reg_font.render("Goal", True, (46, 49, 49))
                    offset = int(min(cell_width, cell_height) / 3.0)
                    goal_text_point = circle_center[0] - font_size, circle_center[1] - font_size/1.5
                    screen.blit(text, goal_text_point)

                # Current state.
                if (i+1,grid_mdp.height - j) == (state.x, state.y) and agent_shape is None:
                    tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 4)

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
