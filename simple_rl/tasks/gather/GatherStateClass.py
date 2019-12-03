''' GatherStateClass.py: Contains the GatherState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State
import numpy as np
import sys
import time
import matplotlib

if sys.platform == "darwin":
    matplotlib.use('MacOSX')

COLORS = {
    'agent1': (0, 34, 244),
    'agent2': (236, 51, 35),
    'orientation': (46, 47, 46),
    'apple': (132, 249, 77),
    'light': (140, 139, 42),
    'walls': (138, 140, 137),
}

class GatherState(State):

    def __init__(self, agent1, agent2, apple_locations, apple_times, render_time=0.01):
        super(GatherState, self).__init__(data=[], is_terminal=False)

        # Locations of player 1 and player 2
        self.agent1, self.agent2 = agent1, agent2
        self.apple_locations = apple_locations
        # TODO: 1. Incorporate apple_times into hash, str, and eq
        self.apple_times = apple_times
        self.x_dim = apple_locations.shape[0]
        self.y_dim = apple_locations.shape[1]
        self.render_time = render_time

    def __hash__(self):
        return hash(tuple(str(self.agent1), str(self.agent2), str(self.apple_locations)))

    def __str__(self):
        stateString = [str(self.agent1), str(self.agent2), self.apple_locations.tostring()]
        return ''.join(stateString)

    def __eq__(self, other):
        if not isinstance(other, GatherState):
            return False
        return self.agent1 == other.agent1 and self.agent2 == other.agent2 and np.array_equal(self.apple_locations, other.apple_locations)

    def to_rgb(self):
        # 3 by x_length by y_length array with values 0 (0) --> 1 (255)
        board = np.zeros(shape=[3, self.x_dim, self.y_dim])

        # Orientation (do this first so that more important things override)
        orientation = self.agent1.get_orientation()
        board[:, orientation[0], orientation[1]] = COLORS['orientation']
        orientation = self.agent2.get_orientation()
        board[:, orientation[0], orientation[1]] = COLORS['orientation']

        # Beams
        if self.agent1.is_shining:
            beam = self.agent1.get_beam(self.x_dim, self.y_dim)
            board[:, beam[0], beam[1]] = np.transpose(np.ones(shape=[beam[2], 1])*COLORS['light'])
        if self.agent2.is_shining:
            beam = self.agent2.get_beam(self.x_dim, self.y_dim)
            board[:, beam[0], beam[1]] = np.transpose(np.ones(shape=[beam[2], 1])*COLORS['light'])

        # Apples
        board[0, (self.apple_locations == 1)] = COLORS['apple'][0]
        board[1, (self.apple_locations == 1)] = COLORS['apple'][1]
        board[2, (self.apple_locations == 1)] = COLORS['apple'][2]

        # Agents
        board[:, self.agent1.x, self.agent1.y] = COLORS['agent1']
        board[:, self.agent2.x, self.agent2.y] = COLORS['agent2']

        # Walls
        board[:, np.arange(0, self.x_dim), 0] = np.transpose(np.ones(shape=[self.x_dim, 1])*COLORS['walls'])
        board[:, np.arange(0, self.x_dim), self.y_dim - 1] = np.transpose(np.ones(shape=[self.x_dim, 1])*COLORS['walls'])
        board[:, 0, np.arange(0, self.y_dim)] = np.transpose(np.ones(shape=[self.y_dim, 1])*COLORS['walls'])
        board[:, self.x_dim - 1, np.arange(0, self.y_dim)] = np.transpose(np.ones(shape=[self.y_dim, 1])*COLORS['walls'])
        board = board/(255.0)

        return np.transpose(board, axes=[2, 1, 0])

    def generate_next_state(self):
        # assume that we are just copying the current apple locations
        # print self.apple_locations
        # new_apple_locations = np.copyto(np.empty_like(self.apple_locations), self.apple_locations)
        new_apple_locations = np.array(self.apple_locations)
        new_apple_times = {}
        for apple in self.apple_times.keys():
            new_apple_times[apple] = self.apple_times[apple]
        return GatherState(self.agent1, self.agent2, new_apple_locations, new_apple_times)

    def show(self):
        rgb = self.to_rgb()
        matplotlib.pyplot.imshow(rgb)
        matplotlib.pyplot.pause(self.render_time)
        matplotlib.pyplot.draw()

class GatherAgent():

    def __init__(self, x, y, is_shining, orientation, hits, frozen_time_remaining):
        self.x, self.y, self.is_shining, = x, y, is_shining
        self.orientation, self.hits = orientation, hits
        self.frozen_time_remaining = frozen_time_remaining

    def get_orientation(self):
        if self.orientation == 'NORTH':
            return self.x, self.y - 1
        if self.orientation == 'SOUTH':
            return self.x, self.y + 1
        if self.orientation == 'WEST':
            return self.x - 1, self.y
        if self.orientation == 'EAST':
            return self.x + 1, self.y
        assert False, 'Invalid direction.'

    def get_beam(self, x_dim, y_dim):
        assert self.is_shining, 'get_beam called when beam not shining'
        orientation = self.get_orientation()
        if self.orientation == 'NORTH':
            return orientation[0], np.arange(0, orientation[1] + 1), orientation[1] + 1
        if self.orientation == 'SOUTH':
            return orientation[0], np.arange(orientation[1], y_dim), y_dim - orientation[1]
        if self.orientation == 'WEST':
            return np.arange(0, orientation[0] + 1), orientation[1], orientation[0] + 1
        if self.orientation == 'EAST':
            return np.arange(orientation[0], x_dim), orientation[1], x_dim - orientation[0]
        assert False, 'Invalid direction.'

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        agentString = ['{:02d}'.format(self.x), '{:02d}'.format(self.y), '1' if self.is_shining else '0', self.orientation, str(self.hits), str(self.frozen_time_remaining)]
        return ''.join(agentString)

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return str(self) == str(other)

    def clone(self):
        return GatherAgent(self.x, self.y, self.is_shining, self.orientation, self.hits, self.frozen_time_remaining)


if __name__ == '__main__':
    agent1 = GatherAgent(32, 6, False, 'NORTH', 0, 0)
    agent2 = GatherAgent(31, 5, False, 'NORTH', 0, 0)
    agent3 = GatherAgent(5, 6, True, 'NORTH', None, None)
    agent4 = GatherAgent(1, 2, True, 'EAST', None, None)
    state1 = GatherState(agent1, agent2, np.zeros(shape=[35, 11], dtype=np.int32))
    state2 = GatherState(agent3, agent4, np.zeros(shape=[21, 11], dtype=np.int32))
    state3 = GatherState(agent3, agent4, np.zeros(shape=[21, 11], dtype=np.int32))
    matplotlib.pyplot.imshow(state1.to_rgb())
    matplotlib.pyplot.show()
