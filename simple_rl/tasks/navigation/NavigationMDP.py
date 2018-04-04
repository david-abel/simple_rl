''' NavigationMDP.py: Contains the NavigationMDP class. '''

# Python imports.
from __future__ import print_function
import numpy as np
from simple_rl.tasks import GridWorldMDP

class NavigationMDP(GridWorldMDP):
    
    '''
        Class for Navigation MDP from:
            MacGlashan, James, and Michael L. Littman. "Between Imitation and Intention Learning." IJCAI. 2015.
    '''

    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, 
                 width=30, 
                 height=30, 
                 init_loc=(1,1),
                 rand_init=True,
                 goal_locs=[(21, 21)],
                 cell_types=["empty", "yellow", "red", "green", "purple"],
                 cell_rewards=[0, 0, -10, -10, -10],
                 gamma=0.99,
                 slip_prob=0.00,
                 step_cost=0.0,
                 goal_reward=1.0,
                 is_goal_terminal=True,
                 init_state=None,
                 name="Navigation MDP"):
        """
        Note: 1. locations and state dimensions start from 1 instead of 0. 
              2. 2d locations are interpreted in (x,y) format.
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
            cell_type (list of cell types: [str, str, ...]): non-goal cell types
            cell_rewards (reward mapping for each cell type: [int, int, ...]): reward value for cells in @cell_type
        """
        
        assert height > 0 and isinstance(height, int) and width > 0 and isinstance(width, int), "height and widht must be integers and > 0"
        
        # Probability of each cell type
        vacancy_prob = 0.8
        self.cell_prob = [4.*vacancy_prob/5., vacancy_prob/5.] + [(1-vacancy_prob)/3.] * 3
        # Matrix for identifying cell type and associated reward
        self.cells = np.random.choice(len(cell_types), p=self.cell_prob, size=height*width).reshape(height,width)
        self.cell_rewards = np.asarray([[cell_rewards[item] for item in row] for row in self.cells]).reshape(height,width)
        self.goal_reward = goal_reward
        
        GridWorldMDP.__init__(self, 
                            width=width, 
                            height=height,
                            init_loc=init_loc,
                            rand_init=rand_init,
                            goal_locs=goal_locs,
                            lava_locs=[()],
                            walls=[], # no walls in this mdp
                            is_goal_terminal=is_goal_terminal,
                            gamma=gamma,
                            init_state=init_state,
                            slip_prob=slip_prob,
                            step_cost=step_cost,
                            name=name)
    
    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if self._is_goal_state_action(state, action):
            return self.goal_reward - self.step_cost
        elif self.cell_rewards[state.x-1, state.y-1] == 0:
            return 0 - self.step_cost
        else:
            return self.cell_rewards[state.x-1, state.y-1]
        
    def visualize_grid(self):
        import matplotlib.pyplot as plt
        from matplotlib import colors
        cmap = colors.ListedColormap(['white','yellow','red','lime','magenta'])
        
        plt.imshow(self.cells[:,::-1].transpose(1,0), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.show()
