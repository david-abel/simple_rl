# Multi-target search extension of maze2d.
import pygame
import math
import numpy as np
import random
import cv2
from maze2d import GridWorld, Environment, dist, inclusive_within, lighter


class MultiTargetGridWorld(GridWorld):
    """The only difference with GridWorld is that there are multiple targets
    in MultiTargetGridWorld."""

    def __init__(self, worldstr):
        super().__init__(worldstr)

    def _interpret(self, worldstr):
        lines = [l for l in worldstr.splitlines()
             if len(l) > 0]
        w, h = len(lines[0]), len(lines)
        arr2d = np.zeros((w,h), dtype=np.int32)
        robot_pose = None
        landmarks = []
        target_poses = []
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
                    landmarks.append((x,y))
                elif c == "T":
                    arr2d[x,y] = 2
                    target_poses.append((x,y))
                elif c == "R":
                    arr2d[x,y] = 0
                    robot_pose = (x,y,0)
        if robot_pose is None:
            raise ValueError("No initial robot pose!")
        if target_poses is None:
            raise ValueError("No target!")
        self._d = arr2d
        self._robot_pose = robot_pose
        self._landmarks = landmarks
        self._target_poses = target_poses
        self._last_z = []
        self._last_belief = None

    @property
    def target_pose(self):
        raise NotImplemented

    @property
    def target_poses(self):
        return self._target_poses

    def update_belief(self, belief, first_target_id):
        """belief is oopomdp belief state"""
        self._last_belief = {}
        for objid in belief:
            target_index = objid - first_target_id
            self._last_belief[target_index] = belief.get_distribution(objid)

    # Observe multiple targets
    def if_observe_at(self, pose, sensor_params, **kwargs):
        def in_field_of_view(th, view_angles):
            """Determines if the beame at angle `th` is in a field of view of size `view_angles`.
            For example, the view_angles=180, means the range scanner scans 180 degrees
            in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
            fov_right = (0, view_angles / 2)
            fov_left = (2*math.pi - view_angles/2, 2*math.pi)
            return inclusive_within(th, fov_left) or inclusive_within(th, fov_right)

        known_correspondence = kwargs.get('known_correspondence', False)
        target_poses = kwargs.get('target_poses', None)

        if target_poses is None:
            target_poses = self._target_poses

        # Only observe the target
        rx, ry, rth = pose
        z_candidates = [(dist(l, (rx, ry)),  # distance
                         (math.atan2(l[1] - ry, l[0] - rx) - rth) % (2*math.pi), # bearing (i.e. orientation)
                         i)
                        for i,l in enumerate(target_poses)  #get_coords(self._d)
                        if dist(l, pose[:2]) <= sensor_params['max_range']\
                        and dist(l, pose[:2]) >= sensor_params['min_range']]        
        z_withc = [(d, th, i) for d,th,i in z_candidates
                   if in_field_of_view(th, sensor_params['view_angles'])]
        z = [(d, th) for d, th, _ in z_withc]
        c = [i for _, _, i in z_withc]

        if known_correspondence:
            return z, c
        else:
            return z

    def update_observation(self, observation):
        z, c = observation
        self._last_z = z

    def observable_at(self, pose, sensor_params, **kwargs):
        target_index = kwargs.get("target_index", [])
        z, c = self.if_observe_at(pose, sensor_params, known_correspondence=True, **kwargs)
        z_by_c = {c[j]:z[j] for j in range(len(c))}
        result = {}
        if target_index in z_by_c:
            return True
        else:
            return False


class MultiTargetEnvironment(Environment):
    def __init__(self, gridworld,
                 sensor_params={
                     'max_range':12,
                     'min_range':1,
                     'view_angles': math.pi,
                     'sigma_dist': 0.01,
                     'sigma_bearing': 0.01},
                 res=30, fps=30, controllable=True):                     
        super().__init__(gridworld, sensor_params=sensor_params,
                         res=res, fps=fps, controllable=controllable)
        # Generate some colors, one per target
        colors = []
        for i in range(len(gridworld.target_poses)):
            colors.append((random.randint(50, 255),
                           random.randint(50, 255),
                           random.randint(50, 255)))
        self._target_colors = colors

    @staticmethod
    def draw_belief(img, belief, r, size, target_colors):
        """belief is a mapping from target index to belief distribution."""
        radius = int(round(r / 2))

        circle_drawn = {}  # map from pose to number of times drawn

        for target_index in belief:
            hist = belief[target_index].histogram
            color = target_colors[target_index]

            last_val = -1
            # just display top 5
            count = 0
            for state in reversed(sorted(hist, key=hist.get)):
                if state.objclass == 'target':
                    if last_val != -1:
                        color = lighter(color, 1-hist[state]/last_val)
                    if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.99:
                        tx, ty = state['pose']
                        if (tx,ty) not in circle_drawn:
                            circle_drawn[(tx,ty)] = 0
                        circle_drawn[(tx,ty)] += 1
                        
                        cv2.circle(img, (ty*r+radius,
                                         tx*r+radius), size//circle_drawn[(tx,ty)], color, thickness=-1)
                        last_val = hist[state]

                        
                        count +=1
                        if count >= 20:
                            break
                    
    def render_env(self, display_surf):
        # draw robot, a circle and a vector
        rx, ry, rth = self._gridworld.robot_pose
        r = self._resolution  # Not radius!
        img = np.copy(self._img)
        if self._gridworld.last_belief is not None:
            MultiTargetEnvironment.draw_belief(img, self._gridworld.last_belief, r, r//3, self._target_colors)
        Environment.draw_observation(img, self._gridworld.last_observation, rx, ry,
                                     rth, r, r//4, color=(12,12,255))
        Environment.draw_robot(img, rx*r, ry*r, rth, r, color=(255, 12, 12))
        pygame.surfarray.blit_array(display_surf, img)

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
            elif event.key == pygame.K_d:
                print("DETECT!")
                action = "detect"
            if self._controllable:                    
                u = self._gridworld.move_robot(*Environment.ACTIONS[action])  # rotate left 45 degree                    
                if u is not None:
                    z_withc = self._gridworld.provide_observation(self._sensor_params,
                                                                  known_correspondence=True)
                print("     action: %s" % str(u))
                print("observation: %s" % str(z_withc))
                print("------------")
            return action
        
        


world1 = \
"""
Rx.....
.x..T..
.......
.T.....
.......
.....T.
.......
"""    

def unittest():
    gridworld = MultiTargetGridWorld(world1)
    env = MultiTargetEnvironment(gridworld,
                                 sensor_params={
                                     'max_range':5,
                                     'min_range':1,
                                     'view_angles': math.pi/2,
                                     'sigma_dist': 0.01,
                                     'sigma_bearing': 0.01},
                                 res=30, fps=10, controllable=True)
    env.on_execute()

if __name__ == '__main__':
    unittest()
