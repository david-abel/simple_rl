# Test POMCP
from simple_rl.planning.POMCPClass import POMCP
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

# Simple gridworld POMDP. The goal is to reach target location in a 2d gridworld.
# The robot has a fan-shaped observation function that tells the robot
# the location of road_blocks and target in the field of view.

world0 = \
"""
R..
...
..T
"""

world1 = \
"""
R..x.
...x.
....T
"""

def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def inclusive_within(v, rg):
    a, b = rg # range
    return v >= a and v <= b

class GridWorld:
    """THIS IS ADAPTED FROM MY slam_exercise PROJECT"""
    
    def __init__(self, worldstr):
        lines = [l for l in worldstr.splitlines()
             if len(l) > 0]
        w, h = len(lines[0]), len(lines)
        arr2d = np.zeros((w,h), dtype=np.int32)
        robot_pose = None
        landmarks = set({})
        target_pose = None
        for y, l in enumerate(lines):
            if len(l) != w:
                raise ValueError("World size inconsistent."\
                                 "Expected width: %d; Actual Width: %d"
                                 % (w, len(l)))
            for x, c in enumerate(l):
                if c == ".":
                    arr2d[x,y] = 0
                elif c == "x":
                    arr2d[x,y] = 1
                    landmarks.add((x,y))
                elif c == "T":
                    arr2d[x,y] = 2
                    target_pose = (x,y)
                    landmarks.add((x,y))
                elif c == "R":
                    arr2d[x,y] = 0
                    robot_pose = (x,y,0)
        if robot_pose is None:
            raise ValueError("No initial robot pose!")
        if target_pose is None:
            raise ValueError("No target!")
        self._d = arr2d
        self._robot_pose = robot_pose
        self._landmarks = landmarks
        self._target_pose = target_pose
        self._last_z = []
        
    @property
    def width(self):
        return self._d.shape[0]
    
    @property
    def height(self):
        return self._d.shape[1]

    @property
    def arr(self):
        return self._d

    @property
    def last_observation(self):
        return self._last_z

    @property
    def robot_pose(self):
        # The pose is (x,y,th)
        return self._robot_pose

    @property
    def target_pose(self):
        # The pose is (x,y,th)
        return self._target_pose    

    def valid_pose(self, x, y):
        if x >= 0 and x < self.width \
           and y >= 0 and y < self.height:
            return self._d[x,y] == 0 or self._d[x,y] == 2  # free, or target.
        return False

    def if_move_by(self, pose, forward, angle):
        rx, ry, rth = pose
        rth += angle
        rx = int(round(rx + forward*math.cos(rth)))
        ry = int(round(ry + forward*math.sin(rth)))
        rth = rth % (2*math.pi)
        if self.valid_pose(rx, ry):
            return (rx, ry, rth)
        else:
            return pose

    def move_robot(self, forward, angle):
        """
        forward: translational displacement (vt)
        angle: angular displacement (vw)
        """
        # First turn, then move forward.
        begin_pose = self._robot_pose
        rx, ry, rth = self.if_move_by(self._robot_pose, forward, angle)

        # Odometry motion model
        if (rx, ry, rth) != self._robot_pose:
            self._robot_pose = (rx, ry, rth)
            rx0, ry0, rth0 = begin_pose
            dtrans = dist((rx, ry), (rx0, ry0))
            drot1 = (math.atan2(ry - ry0, rx - rx0) - rth0) % (2*math.pi)
            drot2 = (rth - rth0 - drot1) % (2*math.pi)
            return (drot1, dtrans, drot2)
        else:
            return (0, 0, 0)

    def if_observe_at(self, pose, sensor_params, known_correspondence=False):
        def in_field_of_view(th, view_angles):
            """Determines if the beame at angle `th` is in a field of view of size `view_angles`.
            For example, the view_angles=180, means the range scanner scans 180 degrees
            in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
            fov_right = (0, view_angles / 2)
            fov_left = (2*math.pi - view_angles/2, 2*math.pi)
            return inclusive_within(th, fov_left) or inclusive_within(th, fov_right) 
        rx, ry, rth = pose
        z_candidates = [(dist(l, (rx, ry)),  # distance
                         (math.atan2(l[1] - ry, l[0] - rx) - rth) % (2*math.pi), # bearing (i.e. orientation)
                         i)
                        for i,l in enumerate(sorted(self._landmarks))  #get_coords(self._d)
                        if dist(l, pose[:2]) <= sensor_params['max_range']\
                        and dist(l, pose[:2]) >= sensor_params['min_range']]

        z_withc = [(d, th, i) for d,th,i in z_candidates
                   if in_field_of_view(th, sensor_params['view_angles'])]
        z = [(d, th) for d, th, _ in z_withc]
        c = [i for _, _, i in z_withc]
        self._last_z = z

        if known_correspondence:
            return z, c
        else:
            return z        

    def provide_observation(self, sensor_params, known_correspondence=False):
        """Given the current robot pose, provide the observation z."""
            
        # RANGE BEARING
        # TODO: right now the laser penetrates through obstacles. Fix this?
        return self.if_observe_at(self._robot_pose, sensor_params)


class Environment:
    """THIS IS ADAPTED FROM MY slam_exercise PROJECT"""

    def __init__(self, gridworld,
                 sensor_params={
                     'max_range':12,
                     'min_range':1,
                     'view_angles': math.pi,
                     'sigma_dist': 0.01,
                     'sigma_bearing': 0.01},
                 res=30, fps=30, controllable=True):
        """
        r: resolution, number of pixels per cell width.
        """
        self._gridworld = gridworld
        self._resolution = res
        self._sensor_params = sensor_params
        self._controllable = controllable
        self._img = self._make_gridworld_image(res)
        
        
        self._running = True
        self._display_surf = None
        self._fps = fps
        self._playtime = 0

    def _make_gridworld_image(self, r):
        arr2d = self._gridworld.arr
        w, h = arr2d.shape
        img = np.full((w*r,h*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(h):
                if arr2d[x,y] == 0:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (255, 255, 255), -1)
                elif arr2d[x,y] == 1:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                elif arr2d[x,y] == 2:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (255, 165, 0), -1)                    
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)                    
        return img

    @staticmethod
    def draw_robot(img, x, y, th, size, color=(255,12,12)):
        radius = int(round(size / 2))
        cv2.circle(img, (y+radius, x+radius), radius, color, thickness=2)

        endpoint = (y+radius + int(round(radius*math.sin(th))),
                    x+radius + int(round(radius*math.cos(th))))
        cv2.line(img, (y+radius,x+radius), endpoint, color, 2)

    @staticmethod
    def draw_observation(img, z, rx, ry, rth, r, size, color=(12,12,255)):
        radius = int(round(r / 2))
        for d, th in z:
            lx = rx + int(round(d * math.cos(rth + th)))
            ly = ry + int(round(d * math.sin(rth + th)))
            cv2.circle(img, (ly*r+radius,
                             lx*r+radius), size, (12, 12, 255), thickness=-1)

    def render_env(self, display_surf):
        # draw robot, a circle and a vector
        rx, ry, rth = self._gridworld.robot_pose
        r = self._resolution  # Not radius!
        img = np.copy(self._img)
        Environment.draw_robot(img, rx*r, ry*r, rth, r, color=(255, 12, 12))
        Environment.draw_observation(img, self._gridworld.last_observation, rx, ry, rth, r, r//3, color=(12,12,255))
        pygame.surfarray.blit_array(display_surf, img)
        
    @property
    def img_width(self):
        return self._img.shape[0]
    
    @property
    def img_height(self):
        return self._img.shape[1]

    @property
    def sensor_params(self):
        return self._sensor_params
 
    def on_init(self):
        # pygame init
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            u = None
            if self._controllable:
                if event.key == pygame.K_LEFT:
                    u = self._gridworld.move_robot(0, -math.pi/4)  # rotate left 45 degree
                elif event.key == pygame.K_RIGHT:
                    u = self._gridworld.move_robot(0, math.pi/4)  # rotate left 45 degree
                elif event.key == pygame.K_UP:
                    u = self._gridworld.move_robot(1, 0)

                if u is not None:
                    z_withc = self._gridworld.provide_observation(self._sensor_params,
                                                                  known_correspondence=True)
                print("     action: %s" % str(u))
                print("observation: %s" % str(z_withc))
                print("------------")

                
    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0
    
    def on_render(self):
        # self._display_surf.blit(self._background, (0, 0))
        self.render_env(self._display_surf)
        
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        rx, ry, rth = self._gridworld.robot_pose
        pygame.display.set_caption("(%.2f,%.2f,%.2f) %s" % (rx, ry, rth*180/math.pi,
                                                            fps_text))
        pygame.display.flip() 
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


class Maze2D_State(State):
    def __init__(self, robot_pose, target_pose):
        self.robot_pose = robot_pose
        self.target_pose = target_pose
        super().__init__(data=[robot_pose, target_pose])
    def unpack(self):
        return self.robot_pose, self.target_pose
    def __str__(self):
        return 'Maze2D_State::{}'.format(self.data)
    def __repr__(self):
        return self.__str__()


class Maze2D_BeliefDistribution:
    def __init__(self, maze2dpomdp):
        self._maze2dpomdp = maze2dpomdp
        self._distribution_target_pose = defaultdict(lambda: 1)

    def __getitem__(self, state):
        """We assume that the agent knows its own pose. Therefore, if the robot_pose
        in `state` is not equal to the gridworld's robot pose, then return 0. Otherwise,
        return a constant (we're not taking care of the normalization here). If the state
        has incorrect 'object_detected' then return 0 as well."""
        if state.robot_pose != self._maze2dpomdp.gridworld.robot_pose:
            return 0
        else:
            return self._distribution_target_pose[state.target_pose] # just a constant

    def __setitem__(self, state, value):
        _, target_pose = state.unpack()
        self._distribution_target_pose[target_pose] = value

    def __len__(self):
        """This could be a huge number"""
        gridworld = self._maze2dpomdp.gridworld
        w, h = gridworld.width, gridworld.height
        dof_orientation = 7 # 45-degree increment
        return ((w*h)*dof_orientation**3) * (w*h)

    def __str__(self):
        return "MOS3D_BeliefDistribution(robot_pose:%s)" % str(self._maze2dpomdp.gridworld.robot_pose)

    def __hash__(self):
        keys = tuple(self._distribution_target_pose.keys())
        return hash(keys)

    def __eq__(self):
        if isinstance(other, BeliefDistribution):
            return self._distribution_target_pose == other._distribution_target_pose
        return False

    def mpe(self):
        """Most Probable Explanation; i.e. the state with the highest probability"""
        robot_pose = self._maze2dpomdp.gridworld.robot_pose
        target_pose = None
        if len(self._distribution_target_pose) > 0:
            pose = max(self._distribution_target_pose,
                       key=self._distribution_target_pose.get,
                       default=random.choice(list(self._distribution_target_pose)))
            if self._distribution_target_pose[target_pose] > 1:
                target_pose = pose

        if target_pose is None:
            # No state has been associated with a probability. So they all have
            # the same. Thus generate a random state for the objects.
            # target_pose = GridWorld.target_pose_to_tuple({
            #     objid:(random.randint(0, self.gridworld.width-1),
            #            random.randint(0, self.gridworld.length-1),
            #            random.randint(0, self.gridworld.height-1))
            #     for objid in self.gridworld.objects
            # })
            target_pose = self.gridworld.target_pose
        return Maze2D_State(robot_pose, target_pose)

    @property
    def gridworld(self):
        return self._maze2dpomdp.gridworld
    

class Maze2D_BeliefState(BeliefState):
    def __init__(self, maze2dpomdp, distribution=None):
        if distribution is None:
            super().__init__(Maze2D_BeliefDistribution(maze2dpomdp))
        else:
            super().__init__(distribution)

    @property
    def gridworld(self):
        return self.distribution.gridworld

    def sample(self, sampling_method='random'):
        if sampling_method == 'random':  # random uniform
            robot_pose = self.gridworld.robot_pose
            # target_pose = {
            #     objid:(random.randint(0, self.gridworld.width-1),
            #            random.randint(0, self.gridworld.length-1),
            #            random.randint(0, self.gridworld.height-1))
            #     for objid in self.gridworld.objects
            # }
            target_pose = self.gridworld.target_pose
            return Maze2D_State(robot_pose, target_pose)
        elif sampling_method == 'max':
            return self.distribution.mpe()
        raise NotImplementedError('Sampling method {} not implemented yet'.format(sampling_method))            
        

class Maze2DPOMDP(POMDP):
    ACTIONS = {
        0:(1, 0),  # forward
        1:(-1, 0), # backward
        2:(0, -math.pi/4),  # left 45 degree
        3:(0, math.pi/4)    # right 45 degree
    }

    def __init__(self, gridworld, sensor_params):
        self._gridworld = gridworld
        self._sensor_params = sensor_params
        b0 = Maze2D_BeliefState(self)
        super().__init__(list(Maze2DPOMDP.ACTIONS.keys()),
                         [],  # This list of observations was never used anywhere in simple_rl;
                              # It's also impossible to provide this list for our problem.
                         self._transition_func,
                         self._reward_func,
                         self._observation_func,
                         b0,
                         belief_updater_type=None)  # forget about belief updater.

    @property
    def gridworld(self):
        return self._gridworld

    def _transition_func(self, state, action):
        state_robot = state.robot_pose
        action = Maze2DPOMDP.ACTIONS[action]
        next_state_robot = self.gridworld.if_move_by(state_robot, action[0], action[1])
        next_state_target = state.target_pose
        return Maze2D_State(next_state_robot, next_state_target)

    def _observation_func(self, next_state, action):
        next_state_robot = next_state.robot_pose
        observation = tuple(map(tuple, self.gridworld.if_observe_at(next_state_robot, self._sensor_params, known_correspondence=True)))
        return observation
    
    def _reward_func(self, state, action, next_state):
        next_state_robot = next_state.robot_pose
        rx, ry, rth = next_state_robot
        observation = self.gridworld.if_observe_at(next_state_robot, self._sensor_params, known_correspondence=True)
        reward = 0
        # for d, th in observation[0]:
        #     lx = rx + int(round(d * math.cos(rth + th)))
        #     ly = ry + int(round(d * math.sin(rth + th)))
        #     if self.gridworld.target_pose == (lx,ly):
        #         reward += 0.5
        if self.gridworld.target_pose == (rx, ry):
            reward += 10
        return reward

    def execute_agent_action(self, action):
        """Completely overriding parent's function. There is no belief update here."""
        # TODO: cur_state IS WRONG! NOT a POMDP        
        self.cur_state = Maze2D_State(self._gridworld.robot_pose, self._gridworld.target_pose)
        next_state = self.transition_func(self.cur_state, action)
        observation = self.observation_func(next_state, action)
        reward = self.reward_func(self.cur_state, action, next_state)
        self._gridworld.move_robot(*Maze2DPOMDP.ACTIONS[action])
        return reward, observation

    def is_in_goal_state(self):
        return self.cur_state.robot_pose[:2] == self.gridworld.target_pose


class Experiment:

    def __init__(self, env, pomdp, planner, render=True):
        self._env = env
        self._planner = planner
        self._pomdp = pomdp
        self._discounted_sum_rewards = 0

    def run(self):
        if self._env.on_init() == False:
            raise Exception("Environment failed to initialize")

        num_iter = 0
        # self._env.on_loop()
        self._env.on_render()

        try:
            while self._env._running and not self._pomdp.is_in_goal_state():
                # for event in pygame.event.get():
                #     self._env.on_event(event)
                action, reward, observation = \
                    self._planner.plan_and_execute_next_action()  # the action is a control to the robot
                print("%d: Action: %s; Reward: %.3f; Observation: %s"
                      % (num_iter, str(action), reward, str(observation)))
                self._discounted_sum_rewards += ((self._planner.gamma ** num_iter) * reward)

                # self._env._last_observation = self._pomdp.gridworld.provide_observation()
                self._env.on_loop()
                self._env.on_render()
                num_iter += 1
        except KeyboardInterrupt:
            print("Stopped")
            self._env.on_cleanup()
            return
            
        print("Done!")
        time.sleep(1)
        self._env.on_cleanup()

def _rollout_policy(tree, actions):
    return random.choice(actions)
    

def main():
    gridworld = GridWorld(world1)
    sys.stdout.write("Initializing POMDP...")
    env = Environment(gridworld,
                      sensor_params={
                          'max_range':3,
                          'min_range':1,
                          'view_angles': math.pi/2,
                          'sigma_dist': 0.01,
                          'sigma_bearing': 0.01},
                      res=30, fps=10, controllable=False)
    # env.on_execute()
    sys.stdout.write("done\n")
    sys.stdout.write("Initializing Planner: POMCP...")
    pomdp = Maze2DPOMDP(gridworld, env.sensor_params)
    sys.stdout.write("Initializing Planner: POMCP...")    
    planner = POMCP(pomdp, max_time=3., max_depth=5, gamma=0.99, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(2))
    sys.stdout.write("done\n")    
    experiment = Experiment(env, pomdp, planner)
    print("Running experiment")
    experiment.run()


if __name__ == '__main__':
    main()
