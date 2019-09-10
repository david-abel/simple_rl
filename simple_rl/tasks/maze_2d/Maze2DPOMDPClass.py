# Test POMCP on a POMDP
from maze2d import GridWorld, Environment, dist
from simple_rl.pomdp.POMCPClass import POMCP, POMCP_Particles
from simple_rl.pomdp.POMDPClass import POMDP
from simple_rl.pomdp.shared import BeliefState
from simple_rl.mdp.StateClass import State

import pygame
import cv2
import math
import numpy as np
# import numpy.random as random
import random
import sys
import time
from collections import defaultdict

class Maze2D_State(State):
    def __init__(self, robot_pose, target_pose):
        self.robot_pose = robot_pose
        self.target_pose = target_pose
        super().__init__(data=[])  # data is not useful if hash function is overridden.
    def unpack(self):
        return self.robot_pose, self.target_pose
    def __str__(self):
        return 'Maze2D_State::[%s, %s]' % (str(self.robot_pose), str(self.target_pose))
    def __repr__(self):
        return self.__str__()
    def __eq__(self, other):
        return self.target_pose == other.target_pose\
            and self.robot_pose == other.robot_pose
    def __hash__(self):
        return hash((self.robot_pose, self.target_pose))
    def __getitem__(self, index):
        raise NotImplemented
    def __len__(self):
        raise NotImplemented
    

# TODO: Why Belief State class? It's no different from belief distribtion
# Currently, there is a "belief udpater" in simple_rl, which I think is
# also unnecessary, because belief update rule should be specified
# by the kind of belief distribution.
class Maze2D_BeliefState(BeliefState):
    def __init__(self, gridworld, distribution):
        super().__init__(distribution)
        self._gridworld = gridworld
        self._iter_current_state = Maze2D_State(gridworld.robot_pose, (-1,0))

    def sample(self, sampling_method='random'):
        if sampling_method == 'random':  # random uniform
            return self.distribution.random()
        elif sampling_method == 'max':
            return self.distribution.mpe()
        raise NotImplementedError('Sampling method {} not implemented yet'.format(sampling_method))

    def update(self, *params, **kwargs):
        self.distribution = self.distribution.update(*params, **kwargs)

    def __iter__(self):
        # We want to get a robot pose from current distribution, not the current world.
        self._robot_pose = self.distribution.random().robot_pose
        return self

    def __next__(self):
        target_x, target_y = self._iter_current_state.target_pose
        target_x += 1
        if target_x >= self._gridworld.width:
            target_x = 0
            target_y += 1
        if target_y >= self._gridworld.height:
            raise StopIteration
        else:
            self._iter_current_state =  Maze2D_State(self._robot_pose, (target_x, target_y))
            return self._iter_current_state
        

class Maze2D_POMDP(POMDP):
    """Simple gridworld POMDP. The goal is to correctly estimate the target location
    in a 2d gridworld.  The robot receives observations about landmarks and the
    target. This differs from the MDP case, where the goal is to reach the
    target."""
    
    def __init__(self, gridworld, sensor_params, init_belief, init_true_state):
        self._gridworld = gridworld
        self._sensor_params = sensor_params

        actions = list(Environment.ACTIONS.keys())# + ["detect"]
        
        super().__init__(actions,
                         self._transition_func,
                         self._reward_func,
                         self._observation_func,
                         init_belief,
                         init_true_state)
        
        self._mpe_target_pose = self.cur_belief.distribution.mpe().target_pose        

    @property
    def gridworld(self):
        return self._gridworld

    def _transition_func(self, state, action):
        state_robot = state.robot_pose
        if action != "detect":
            action = Environment.ACTIONS[action]
            next_state_robot = self.gridworld.if_move_by(state_robot, action[0], action[1])
        else:
            next_state_robot = state_robot
        next_state_target = state.target_pose  # the targets are static
        return Maze2D_State(next_state_robot, next_state_target)
    
    def _reward_func(self, state, action, next_state):
        """The reward function used to estimate values of states by the planner. That is,
        this reward function is used by the agent internally when planning."""
        # NOTE: This reward function computes reward based on true information & it's used in the 
        # generator for POMCP - which is not acceptable for a real application. But for the purpose
        # of checking whether the planning algorithm works, it is ok to assume that the "generator"
        # is perfect, meaning that the agent somehow learned a perfect model of the reward in the
        # environment. This shouldn't affect the correctness of the planner.
        if next_state.robot_pose == state.robot_pose:
            return -10
        rx, ry, rth = next_state.robot_pose
        observation = self._observation_func(next_state, action)
        if len(observation[0]) > 0:
            d, th = observation[0][0]  # z -> z for the first target (there's only 1 target so just [0])
            target_x = rx + int(round(d * math.cos(rth + th)))
            target_y = ry + int(round(d * math.sin(rth + th)))
            reward = math.exp(-dist((target_x, target_y), self.gridworld.target_pose))
            return reward - 0.05 # there's a reward if observed a target
        else:
            return 0 - 0.05  # no reward if otherwise.

    def _observation_func(self, next_state, action):
        # Same for the observation function; the planning agent shouldn't be able
        # to sample real observations. But, at this stage, let's do what David Silver
        # did and assume there's a perfect observation model of the world on the agent.
        next_state_robot = next_state.robot_pose
        observation = tuple(map(tuple,
                                self.gridworld.if_observe_at(next_state_robot,
                                                             self._sensor_params,
                                                             target_pose=next_state.target_pose,
                                                             known_correspondence=True)))
        return observation

    def execute_agent_action_update_belief(self, real_action, **kwargs):
        """Completely overriding parent's function.
        Belief update NOT done here; see update_belief"""

        def env_reward_func(agent_mpe_state):
            """The function that computes the reward that the environment gives to the robot"""
            reward = math.exp(-dist(agent_mpe_state.target_pose, self.gridworld.target_pose))
            return reward

        cur_true_state = Maze2D_State(self._gridworld.robot_pose, self._gridworld.target_pose)
        next_true_state = self.transition_func(cur_true_state, real_action)
        real_observation = self.observation_func(next_true_state, real_action)

        # Execute the real action, update the belief.
        self._gridworld.move_robot(*Environment.ACTIONS[real_action])
        self.cur_state = next_true_state
        self._update_belief(real_action, real_observation, **kwargs)

        # Pretend that there is a "detect" action after every action the agent takes,
        # so that it communicates its belief to the environment, with which the environment
        # computes a reward.
        cur_mpe_state = Maze2D_State(self._gridworld.robot_pose, self._mpe_target_pose)
        reward = env_reward_func(cur_mpe_state)  # just compute the reward based on 
        
        # Execute action in the gridworld; Modify the underlying true state
        return reward, real_observation

    def _update_belief(self, real_action, real_observation, **kwargs):
        print("updating belief>>>>")
        print(self.cur_belief)
        self.cur_belief.update(real_action, real_observation,
                               self, kwargs['num_particles'])
        print(">>>>")
        print(self.cur_belief)
        self._gridworld.update_belief(self.cur_belief)
        self._mpe_target_pose = self.cur_belief.distribution.mpe().target_pose

    def is_in_goal_state(self):
        return self.cur_belief.distribution.mpe().target_pose == self.gridworld.target_pose

    def add_transform(self, state):
        """Used for particle reinvigoration"""
        target_x = max(0, min(self.gridworld.width-1, state.target_pose[0] + random.randint(-1, 1)))
        target_y = max(0, min(self.gridworld.height-1, state.target_pose[1] + random.randint(-1, 1)))
        state.target_pose = (target_x, target_y)
        

class Experiment:

    def __init__(self, env, pomdp, planner, render=True, max_episodes=100, pilot=False):
        """
        if 'pilot' is True, the user provides action at every episode.
        """
        self._env = env
        self._planner = planner
        self._pomdp = pomdp
        self._discounted_sum_rewards = 0
        self._num_iter = 0
        self._max_episodes = max_episodes
        self._pilot = pilot

    def run(self):
        if self._env.on_init() == False:
            raise Exception("Environment failed to initialize")

        # self._env.on_loop()
        self._num_iter = 0
        self._env.on_render()

        total_time = 0
        rewards = []
        try:
            while self._env._running \
                  and not self._pomdp.is_in_goal_state()\
                  and (self._pilot or self._num_iter < self._max_episodes):

                reward = None
                if self._pilot:
                    for event in pygame.event.get():
                        action = self._env.on_event(event)
                        if action is not None:
                            action, reward, observation = self._planner.execute_next_action(action)
                            self._env._gridworld.update_observation(observation)
                else:
                    start_time = time.time()
                    action, reward, observation = \
                        self._planner.plan_and_execute_next_action()  # the action is a control to the robot
                    total_time += time.time() - start_time
                    self._env._gridworld.update_observation(observation)

                if reward is not None:
                    print("---------------------------------------------")
                    print("%d: Action: %s; Reward: %.3f; Observation: %s"
                          % (self._num_iter, str(action), reward, str(observation)))
                    print("---------------------------------------------")
                    self._discounted_sum_rewards += ((self._planner.gamma ** self._num_iter) * reward)
                    rewards.append(reward)

                # self._env._last_observation = self._pomdp.gridworld.provide_observation()
                self._env.on_loop()
                self._env.on_render()
                self._num_iter += 1
        except KeyboardInterrupt:
            print("Stopped")
            self._env.on_cleanup()
            return

        print("Done!")
        if self._env._running:
            # Render the final belief
            self._env.on_loop()
            self._env.on_render()
            time.sleep(5)
        self._env.on_cleanup()
        return total_time, rewards

    @property
    def discounted_sum_rewards(self):
        return self._discounted_sum_rewards

    @property
    def num_episode(self):
        return self._num_iter

world0 = \
"""
..........
.R........
..........
..........
..........
..........
.T........
"""

world1 = \
"""
Rx...
.x.xT
.....
"""

world2= \
"""
............................
..xxxxxxxxxxxxxxxxxxxxxxxx..
..xR............T........x..
..x......................x..
..x..xxxxxxxxxxxxxxxxxx..x..
..x..x................x..x..
..x..xxxxxxxxxxxxxxxxxx..x..
..x......................x..
..x......................x..
..xxxxxxxxxxxxxxxxxxxxxxxx..
............................
"""

world3 = \
"""
....................................
....................................
.R..................................
....................................
....................................
.....x.......x.......x.......x......
....................................
....................................
....................................
.T..................................
....................................
....................................
....................................
.....x.......x.......x.......x......
....................................
....................................
....................................
....................................
....................................
""" # textbook

world4 = \
"""
.T........R...
"""

world5 = \
"""
RT
"""


def _rollout_policy(tree, actions):
    return random.choice(actions)

def unittest():
    sensor_params = {
        'max_range':5,
        'min_range':0,
        'view_angles': math.pi/2,
        'sigma_dist': 0.01,
        'sigma_bearing': 0.01
    }
    gridworld = GridWorld(world0)
    sys.stdout.write("Initializing POMDP...")
    env = Environment(gridworld,
                      sensor_params,
                      res=30, fps=10, controllable=False)
    # env.on_execute()
    sys.stdout.write("done\n")
    sys.stdout.write("Initializing Planner: POMCP...")
    num_particles = 5000
    particles = []
    while len(particles) < num_particles:
        random_target_pose = (random.randint(0, gridworld.width-1),
                              random.randint(0, gridworld.height-1))
        particles.append(Maze2D_State(gridworld.robot_pose, random_target_pose))
        sys.stdout.write("Generating particles [%d/%d]\r" % (len(particles), num_particles))
    sys.stdout.write("\n")
    init_belief_distribution = POMCP_Particles(particles)
    init_belief = Maze2D_BeliefState(gridworld, init_belief_distribution)
    init_true_state = Maze2D_State(gridworld.robot_pose, gridworld.target_pose)
    
    pomdp = Maze2D_POMDP(gridworld, env.sensor_params, init_belief, init_true_state)
    sys.stdout.write("Initializing Planner: POMCP...")    
    planner = POMCP(pomdp, num_particles=num_particles,
                    max_time=0.5, max_depth=50, gamma=0.99, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(2))
    sys.stdout.write("done\n")    
    experiment = Experiment(env, pomdp, planner, pilot=False, max_episodes=1000)
    print("Running experiment")
    experiment.run()

if __name__ == '__main__':
    unittest()
