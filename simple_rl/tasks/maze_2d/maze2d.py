# Single-target search. (kaiyu zheng)

import pygame
import cv2
import math
import random
import numpy as np
import moos3d.util as util

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
        self._interpret(worldstr)

    def _interpret(self, worldstr):
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
        self._last_belief = None
        
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
    def last_belief(self):
        return self._last_belief

    def update_belief(self, belief):
        self._last_belief = belief

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

    def if_observe_at(self, pose, sensor_params, **kwargs):
        def in_field_of_view(th, view_angles):
            """Determines if the beame at angle `th` is in a field of view of size `view_angles`.
            For example, the view_angles=180, means the range scanner scans 180 degrees
            in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
            fov_right = (0, view_angles / 2)
            fov_left = (2*math.pi - view_angles/2, 2*math.pi)
            return inclusive_within(th, fov_left) or inclusive_within(th, fov_right)

        known_correspondence = kwargs.get('known_correspondence', False)
        target_pose = kwargs.get('target_pose', None)

        if target_pose is None:
            target_pose = self._target_pose

        # Only observe the target
        rx, ry, rth = pose
        z_candidates = [(dist(l, (rx, ry)),  # distance
                         (math.atan2(l[1] - ry, l[0] - rx) - rth) % (2*math.pi), # bearing (i.e. orientation)
                         i)
                        for i,l in enumerate([target_pose])  #get_coords(self._d)
                        if dist(l, pose[:2]) <= sensor_params['max_range']\
                        and dist(l, pose[:2]) >= sensor_params['min_range']]        
        z_withc = [(d, th, i) for d,th,i in z_candidates
                   if in_field_of_view(th, sensor_params['view_angles'])]
        z = [(d, th) for d, th, _ in z_withc]
        c = [i for _, _, i in z_withc]
        if target_pose == self._target_pose:
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
        controllable (bool) If True, then once the environment receives a keyboard
                            event, such as moving the robot by an arrow key, this
                            action is performed on the gridworld immediately.
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

    def set_controllable(self, val):
        self._controllable = val

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

    @staticmethod
    def draw_belief(img, belief, r, size):
        radius = int(round(r / 2))
        hist = belief.distribution.get_histogram()
        color = (233, 25, 0)
        for state in reversed(sorted(hist, key=hist.get)):
            # when the color is not too light
            if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.95:
                tx, ty = state.target_pose
                cv2.circle(img, (ty*r+radius,
                                 tx*r+radius), size, color, thickness=-1)
                color = util.lighter(color, 0.1)

    def render_env(self, display_surf):
        # draw robot, a circle and a vector
        rx, ry, rth = self._gridworld.robot_pose
        r = self._resolution  # Not radius!
        img = np.copy(self._img)
        if self._gridworld.last_belief is not None:
            Environment.draw_belief(img, self._gridworld.last_belief, r, r//3)
        Environment.draw_observation(img, self._gridworld.last_observation, rx, ry,
                                     rth, r, r//4, color=(12,12,255))
        Environment.draw_robot(img, rx*r, ry*r, rth, r, color=(255, 12, 12))
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

    ACTIONS = {
        0:(1, 0),  # forward
        1:(-1, 0), # backward
        2:(0, -math.pi/4),  # left 45 degree
        3:(0, math.pi/4)    # right 45 degree
    }

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            u = None  # control signal according to motion model
            action = None  # control input by user
            if event.key == pygame.K_LEFT:
                action = 2
            elif event.key == pygame.K_RIGHT:
                action = 3
            elif event.key == pygame.K_UP:
                action = 0
            elif event.key == pygame.K_DOWN:
                action = 1
            if self._controllable:                    
                u = self._gridworld.move_robot(*Environment.ACTIONS[action])  # rotate left 45 degree                    
                if u is not None:
                    z_withc = self._gridworld.provide_observation(self._sensor_params,
                                                                  known_correspondence=True)
                print("     action: %s" % str(u))
                print("observation: %s" % str(z_withc))
                print("------------")
            return action

                
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


world1 = \
"""
Rx...
.x.xT
.....
"""        

def unittest():
    gridworld = GridWorld(world1)
    env = Environment(gridworld,
                      sensor_params={
                          'max_range':3,
                          'min_range':1,
                          'view_angles': math.pi/2,
                          'sigma_dist': 0.01,
                          'sigma_bearing': 0.01},
                      res=30, fps=10, controllable=True)
    env.on_execute()

if __name__ == '__main__':
    unittest()
        
