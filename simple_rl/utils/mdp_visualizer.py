# Python imports.
import sys
try:
    import pygame
    from pygame.locals import *
    pygame.init()
    title_font = pygame.font.SysFont("CMU Serif", 32)
except ImportError:
    print "Error: pygame not installed (needed for visuals)."
    quit()

# Other imports.
from simple_rl.utils.chart_utils import color_ls

def _draw_title_text(mdp, screen):
    scr_width, scr_height = screen.get_width(), screen.get_height()
    title_split = str(mdp).split("_")
    title = title_split[0]
    param_text = " ("
    for param in title_split[1:-1]:
        param_text += param + ", "
    param_text += title_split[-1] + ")"
    formatted_title_text = title[0].upper() + title[1:] + param_text
    title_text = title_font.render(formatted_title_text, True, (46, 49, 49))
    screen.blit(title_text, (scr_width / 2.0 - len(formatted_title_text)*6, scr_width / 20.0))

def _draw_agent_text(agent, screen):
    scr_width, scr_height = screen.get_width(), screen.get_height()
    formatted_agent_text = "agent: " + str(agent)
    agent_text_point = (3*scr_width / 4.0 - len(formatted_agent_text)*6, 18*scr_height / 20.0)
    agent_text = title_font.render(formatted_agent_text, True, (46, 49, 49))
    screen.blit(agent_text, agent_text_point)

def _draw_state_text(state, screen):
    scr_width, scr_height = screen.get_width(), screen.get_height()
    # Clear.
    formatted_state_text = str(state)
    if len(formatted_state_text) > 20:
        # State text is too long, ignore.
        return
    state_text_point = (scr_width / 4.0 - len(formatted_state_text)*6, 18*scr_height / 20.0)
    pygame.draw.rect(screen, (255,255,255), (state_text_point[0] - 20, state_text_point[1]) + (200,40))
    state_text = title_font.render(formatted_state_text, True, (46, 49, 49))
    screen.blit(state_text, state_text_point)

def visualize_policy(mdp, policy, draw_state, action_char_dict, cur_state=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        policy (lambda: S --> A)
        draw_state (lambda)
        action_char_dict (dict):
            Key: action
            Val: str
        cur_state (State)

    Summary:

    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state

    agent_shape = _vis_init(screen, mdp, draw_state, cur_state, value=True)
    draw_state(screen, mdp, cur_state, policy=policy, action_char_dict=action_char_dict, show_value=False, draw_statics=True)

def visualize_value(mdp, draw_state, cur_state=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        draw_state (State)

    Summary:
        Draws the MDP with values labeled on states.
    '''

    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state

    agent_shape = _vis_init(screen, mdp, draw_state, cur_state, value=True)
    draw_state(screen, mdp, cur_state, show_value=True, draw_statics=True)

def visualize_agent(mdp, agent, draw_state, cur_state=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        agent (Agent)
        draw_state (lambda: State --> pygame.Rect)
        cur_state (State)
        scr_width (int)
        scr_height (int)

    Summary:
        Creates a 2d visual of the agent's interactions with the MDP.
    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state
    reward = 0

    agent_shape = _vis_init(screen, mdp, draw_state, cur_state, agent)

    done = False
    while not done:

        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_SPACE:
                # Move agent.
                action = agent.act(cur_state, reward)
                reward, cur_state = mdp.execute_agent_action(action)
                agent_shape = draw_state(screen, mdp, cur_state, agent_shape=agent_shape)

                # Update state text.
                _draw_state_text(cur_state, screen)

        if cur_state.is_terminal():
            # Done! Agent found goal.
            goal_text = "Victory!"
            goal_text_rendered = title_font.render(goal_text, True, (246, 207, 106))
            goal_text_point = scr_width / 2.0 - (len(goal_text)*7), 18*scr_height / 20.0
            screen.blit(goal_text_rendered, goal_text_point)
            done = True

        pygame.display.update()


def _vis_init(screen, mdp, draw_state, cur_state, agent=None, value=False):
    # Pygame setup.
    pygame.init()
    screen.fill((255, 255, 255))
    pygame.display.update()
    done = False

    # Draw name of MDP:
    _draw_title_text(mdp, screen)
    if agent is not None:
        _draw_agent_text(agent, screen)
    if not value:
        # If we're not visualizing the value.
        _draw_state_text(cur_state, screen)
        agent_shape = draw_state(screen, mdp, cur_state, draw_statics=True)

        return agent_shape