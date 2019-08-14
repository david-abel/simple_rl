# Test POMCP on an POMDP  (kaiyu zheng)
from maze2d import GridWorld, Environment, dist
from simple_rl.planning.POMCPClass import POMCP, POMCP_Particles
from simple_rl.pomdp.POMDPClass import POMDP
from simple_rl.mdp.StateClass import State
from simple_rl.pomdp.BeliefStateClass import BeliefState

import pygame
import cv2
import math
import random
import numpy as np
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

    def update(self, *params):
        self.distribution.update(*params)

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
    
    def __init__(self, gridworld, sensor_params, init_belief):
        self._gridworld = gridworld
        self._sensor_params = sensor_params

        init_true_state = Maze2D_State(self._gridworld.robot_pose, self._gridworld.target_pose)
        
        super().__init__(list(Environment.ACTIONS.keys()),
                         self._transition_func,
                         self._reward_func,
                         self._observation_func,
                         init_belief,
                         init_true_state,
                         belief_updater_type=None)  # forget about belief updater.
        
        self._mpe_target_pose = self.cur_belief.distribution.mpe().target_pose        

    @property
    def gridworld(self):
        return self._gridworld

    def _transition_func(self, state, action):
        state_robot = state.robot_pose
        action = Environment.ACTIONS[action]
        next_state_robot = self.gridworld.if_move_by(state_robot, action[0], action[1])
        next_state_target = state.target_pose
        return Maze2D_State(next_state_robot, next_state_target)

    def _observation_func(self, next_state, action):
        next_state_robot = next_state.robot_pose
        observation = tuple(map(tuple,
                                self.gridworld.if_observe_at(next_state_robot,
                                                             self._sensor_params,
                                                             target_pose=next_state.target_pose,
                                                             known_correspondence=True)))
        return observation
    
    def _reward_func(self, state, action, next_state):
        next_state_target = next_state.target_pose
        reward = math.exp(-dist(next_state_target, self.gridworld.target_pose))
        # reward = 0
        # if dist(next_state_target, self.gridworld.target_pose) == 0:#self._mpe_target_pose) == 0:
        #     reward = 1
        # reward += math.exp(-dist(next_state.robot_pose[:2], self._mpe_target_pose))
        return reward

    def execute_agent_action(self, real_action, **kwargs):
        """Completely overriding parent's function.
        Belief update NOT done here; It's done in the planner."""

        cur_true_state = Maze2D_State(self._gridworld.robot_pose, self._gridworld.target_pose)
        next_true_state = self.transition_func(cur_true_state, real_action)
        real_observation = self.observation_func(next_true_state, real_action)

        # Reward is computed based on the belief
        reward = 0
        hist = self.cur_belief.distribution.get_histogram()
        for state in hist:
            next_state = self.transition_func(state, real_action)
            reward += hist[state] * self.reward_func(state, real_action, next_state)

        # Execute action in the gridworld; Modify the underlying true state
        self._gridworld.move_robot(*Environment.ACTIONS[real_action])
        self.cur_state = next_true_state
        return reward, real_observation

    def update_belief(self, real_action, real_observation, **kwargs):
        print("updating belief>>>>")
        print(self.cur_belief)
        self.cur_belief.update(real_action, real_observation, self, kwargs['num_particles'])
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
                else:
                    start_time = time.time()
                    action, reward, observation = \
                        self._planner.plan_and_execute_next_action()  # the action is a control to the robot
                    total_time += time.time() - start_time

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
            time.sleep(2)
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
R......
.......
.......
.......
.......
.....T.
.......
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
    init_belief_distribution = POMCP_Particles(particles)
    init_belief = Maze2D_BeliefState(gridworld, init_belief_distribution)
    
    pomdp = Maze2D_POMDP(gridworld, env.sensor_params, init_belief)
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
