# Python imports.
from __future__ import print_function
from collections import defaultdict
try:
    import pygame
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")
import random
import sys

# Other imports.
from simple_rl.planning import ValueIteration
from simple_rl.tasks import FourRoomMDP
from simple_rl.utils import mdp_visualizer as mdpv


def _draw_state(screen,
                grid_mdp,
                state,
                policy=None,
                action_char_dict={},
                show_value=False,
                agent=None,
                draw_statics=False,
                agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        grid_mdp (MDP)
        state (State)
        show_value (bool)
        agent (Agent): Used to show value, by default uses VI.
        draw_statics (bool)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''
    # Make value dict.
    val_text_dict = defaultdict(lambda : defaultdict(float))
    if show_value:
        if agent is not None:
            # Use agent value estimates.
            for s in agent.q_func.keys():
                val_text_dict[s.x][s.y] = agent.get_value(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(grid_mdp)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.x][s.y] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        vi = ValueIteration(grid_mdp)
        vi.run_vi()
        for s in vi.get_states():
            policy_dict[s.x][s.y] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_locs = grid_mdp.get_goal_locs()
    lava_locs = grid_mdp.get_lava_locs()
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2)

    # Draw the static entities.
    if draw_statics:
        # For each row:
        for i in range(grid_mdp.width):
            # For each column:
            for j in range(grid_mdp.height):

                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                if policy and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    a = policy_dict[i+1][grid_mdp.height - j]
                    if a not in action_char_dict:
                        text_a = a
                    else:
                        text_a = action_char_dict[a]
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered_a, text_center_point)

                if show_value and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i+1][grid_mdp.height - j]
                    color = mdpv.val_to_color(val)
                    pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)

                if grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Draw the walls.
                    top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                    r = pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width-10, cell_height-10), 0)

                if (i+1,grid_mdp.height - j) in goal_locs:
                    # Draw goal.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (154, 195, 157)
                    pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 3.0))

                if (i+1,grid_mdp.height - j) in lava_locs:
                    # Draw goal.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (224, 145, 157)
                    pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 4.0))

                # Current state.
                if not show_value and (i+1,grid_mdp.height - j) == (state.x, state.y) and agent_shape is None:
                    tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 8)

    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255,255,255), agent_shape)
        top_left_point = width_buffer + cell_width*(state.x - 1), height_buffer + cell_height*(grid_mdp.height - state.y)
        tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)

        # Draw new.
        agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 8)

    pygame.display.flip()

    return agent_shape


def _draw_agent(center_point, screen, base_size=20):
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
