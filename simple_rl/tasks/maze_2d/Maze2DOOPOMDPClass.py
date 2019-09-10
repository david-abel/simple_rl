#OO OO
import pygame
from maze2d_oo import MultiTargetGridWorld, MultiTargetEnvironment, dist
from maze2d import Environment

# from simple_rl.pomdp.OOPOMDPClass import OOPOMDP
from simple_rl.mdp.StateClass import State

from simple_rl.pomdp.OOPOMDPClass import OOPOMDP_ObjectState, OOPOMDP, OOPOMDP_State, OOPOMDP_BeliefState
from simple_rl.pomdp.shared import BeliefState, RandomPlanner
from simple_rl.pomdp.OOPOMCPClass import OOPOMCP_Histogram, OOPOMCP
from simple_rl.pomdp.POMCPClass import POMCP, POMCP_Particles

import numpy as np
import random
import math
import sys
import time


class Maze2D_ObjectState(OOPOMDP_ObjectState):
    def __init__(self, objclass, attrs):
        super().__init__(objclass, attrs)

    @property
    def pose(self):
        return self.attributes['pose']


class Maze2D_BeliefState(BeliefState):
    # This is a belief state regarding a SINGLE target, or a robot.
    def __init__(self, objclass, gridworld, distribution):
        super().__init__(distribution)
        self._objclass = objclass
        self._gridworld = gridworld
        self._iter_current_state = Maze2D_ObjectState(self._objclass, (-1,0))

    def sample(self, sampling_method='random'):
        if sampling_method == 'random':  # random uniform
            return self.distribution.random()
        elif sampling_method == 'max':
            return self.distribution.mpe()
        raise NotImplementedError('Sampling method {} not implemented yet'.format(sampling_method))

    def update(self, *params):
        self.distribution = self.distribution.update(*params)

    def __iter__(self):
        return self

    def __next__(self):
        x, y = self._iter_current_state.pose
        x += 1
        if x >= self._gridworld.width:
            x = 0
            y += 1
        if y >= self._gridworld.height:
            raise StopIteration
        else:
            self._iter_current_state =  Maze2D_ObjectState(self._objclass, {'pose':(x,y)})
            return self._iter_current_state



class Maze2D_OOPOMDP(OOPOMDP):
    def __init__(self, gridworld, sensor_params, init_belief, oopomdp_init_state, robot_id, first_target_id):
        self._gridworld = gridworld
        self._sensor_params = sensor_params

        attributes = {"robot": {'pose', 'observed'},
                      "target": {'pose'}}

        domains = {('robot','pose'): lambda object_pose: self._gridworld.valid_pose(object_pose[0],object_pose[1]),
                   ('robot', 'observed'):  # observed is a tuple of observed target's object ids.
                   lambda observed: type(observed) == tuple\
                               and len(observed) <= len(gridworld.target_poses)\
                               and (len(observed) == 0\
                                    or len(observed) != 0\
                                        and max(observed) < first_target_id + len(gridworld.target_poses)\
                                        and min(observed) >= first_target_id),
                   ('target','pose'): lambda object_pose: self._gridworld.valid_pose(object_pose[0],object_pose[1])}

        actions = ["detect"] + list(MultiTargetEnvironment.ACTIONS.keys())
        self._robot_id = robot_id
        self._first_target_id = first_target_id
        self._num_detect_actions = 0

        super().__init__(attributes, domains, actions,
                         self._transition_func,
                         self._reward_func,
                         self._observation_func,
                         init_belief,
                         oopomdp_init_state)


    def _transition_func(self, state, action):
        """
        state (OOPOMDP_State)
        action: control of robot"""
        next_state = state.copy()  # copy the current state so we can modify next_state directly.
        robot_state = state.get_object_state(self._robot_id)
        cur_robot_pose = robot_state['pose']        
        if action != "detect":
            # Move the robot
            action = Environment.ACTIONS[action]
            next_robot_pose = self._gridworld.if_move_by(cur_robot_pose, action[0], action[1])
            next_state.get_object_state(self._robot_id)['pose'] = next_robot_pose
        else: # detect action
            next_robot_pose = cur_robot_pose
            target_poses = []
            for i in range(len(self._gridworld.target_poses)):
                target_id = self._first_target_id + i
                target_state = state.get_object_state(target_id)
                # target_poses.append(target_state['pose'])
                if target_state['pose'] == self._gridworld.target_poses[i]:
                    # Groundtruth-info is used here!!
                    target_poses.append(target_state['pose'])
                    
            observation = tuple(map(tuple,
                                    self._gridworld.if_observe_at(next_robot_pose,
                                                                 self._sensor_params,
                                                                 target_poses=target_poses,
                                                                 known_correspondence=True)))
            rx, ry, rth = cur_robot_pose
            z, c = observation  # raw observation and correspondence
            observed_targets = state.get_object_state(self._robot_id)['observed']
            for j in range(len(z)):
                target_id = self._first_target_id + c[j]
                target_observed = target_id in observed_targets
                if not target_observed:
                    # Now it's detected
                    observed_targets += (target_id,)
            next_state.get_object_state(self._robot_id)['observed'] = tuple(sorted(observed_targets))
        if not self.verify_state(next_state):
            raise ValueError("State transition leads to invalid state!")
        return next_state

    def _observation_func(self, next_state, action):
        """next_state (OOPOMDP_State)"""
        next_robot_state = next_state.get_object_state(self._robot_id)
        next_robot_pose = next_robot_state['pose']

        next_target_poses = []
        for i in range(len(self._gridworld.target_poses)):
            target_id = self._first_target_id + i
            next_target_state = next_state.get_object_state(target_id)
            next_target_poses.append(next_target_state['pose'])
        observation = tuple(map(tuple,
                                self._gridworld.if_observe_at(next_robot_pose,
                                                              self._sensor_params,
                                                              target_poses=next_target_poses,
                                                              known_correspondence=True)))
        return observation

    def _reward_func(self, state, action, next_state):
        """state / next_state (OOPOMDP_State)"""
        # If there are new objects detected in next_state, there's a positive reward.
        reward = 0
        if action == "detect":
            # Need to know which objects the 'detect' concerns - must be objects
            # in the field of view of the robot.
            targets_in_fov = []
            observation = self._observation_func(next_state, action)
            z, c = observation
            if len(z) == 0:
                # if there is a 'detect' when there's no object observed, this is a bad action.
                reward -= 10
            else:
                # observed some objects - they are subjects of detection.
                for j in range(len(z)):
                    target_id = self._first_target_id + c[j]
                    targets_in_fov.append(target_id)

                current_observed_targets = state.get_object_state(self._robot_id)['observed']
                next_observed_targets = next_state.get_object_state(self._robot_id)['observed']

                for objid in targets_in_fov:
                    assert state.get_object_class(objid) == "target"

                    current_observed = objid in current_observed_targets
                    next_observed = objid in next_observed_targets
                    if next_observed is True and current_observed is False:
                        # Correct detection
                        reward += 10
                    elif next_observed is False and current_observed is False:
                        # Incorrect detection
                        reward -= 10
                    elif next_observed is True and current_observed is True:
                        # abusing detect
                        reward -= 2
        else:
            next_robot_pose = next_state.get_object_state(self._robot_id)['pose']
            cur_robot_pose = state.get_object_state(self._robot_id)['pose']
            if next_robot_pose == cur_robot_pose:
                reward -= 10
        return reward - 0.1
    
    def target_index(self, objid):
        """target index in the gridworld"""
        return objid - self._first_target_id

    def observation_model(self, objid, observation,
                          next_object_state,
                          next_robot_state,
                          action):
        """The observation model for object 'objid'"""
        # observation model is equipped on the robot; Assume it's perfect.
        # The observation model is also assumed to be factored by objects;
        robot_pose = next_robot_state['pose']
        target_pose = next_object_state['pose']

        # need to create a "target_poses" list where the target pose of interest
        # is at the correct index
        target_poses = [(float('inf'),float('inf'))]*len(self._gridworld.target_poses)
        target_poses[self.target_index(objid)] = target_pose
        z_target, c_target = self._gridworld.if_observe_at(robot_pose,
                                                           self._sensor_params,
                                                           target_poses=target_poses,
                                                           known_correspondence=True)
        z, c = observation  # may contain multiple targets
        if len(c_target) != 0 and len(c) == 0:
            # We actually observed nothing. So the simulated observation is impossible
            return 0
        sim_obs_by_target = {c_target[j]:z_target[j] for j in range(len(c_target))
                             if c[j] == self.target_index(objid)}
        obs_by_target = {c[j]:z[j] for j in range(len(c))
                         if c[j] == self.target_index(objid)}
        if sim_obs_by_target == obs_by_target:
            return 0.95
        else:
            return 0.05

    def transition_model(self, objid,
                         next_object_state, next_robot_state,
                         object_state, robot_state, action):
        """The transition model for object 'objid'"""
        # transition model is equipped on the robot; Assume it's perfect.
        # The transition model is also assumed to be factored by objects
        cur_robot_pose = robot_state['pose']
        if action != 'detect':
            action = Environment.ACTIONS[action]
            next_robot_pose = self._gridworld.if_move_by(cur_robot_pose, action[0], action[1])
            if next_robot_pose == next_robot_state['pose']:
                if next_object_state['pose'] == object_state['pose']:
                    return 0.99
            return 0.01
        else:
            next_robot_pose = next_robot_state['pose']
            if next_robot_pose != cur_robot_pose:
                return 0.01  # detect action does not move the robot
            cur_target_pose = object_state['pose']
            next_target_pose = next_object_state['pose']
            if cur_target_pose != next_target_pose:
                return 0.01  # detect action does not move the target
            current_observed_targets = robot_state['observed']
            next_observed_targets = next_robot_state['observed']
            
            target_id = objid
            if self._gridworld.observable_at(next_robot_pose, self._sensor_params,
                                             target_index=self.target_index(target_id)):
                if not target_id in next_observed_targets:
                    return 0.05 # observable, but claiming it's not observed next.
                else:
                    return 0.95  # observed next
            else:
                if target_id in next_observed_targets:
                    return 0.05  # not observable, but claiming it's observed next.
                else:
                    if target_id in current_observed_targets:
                        return 0.05  # Impossible to go from observable to not observable
                    else:
                        return 0.95  # doesn't observe now and doesn't observe next
                    
    def execute_agent_action_update_belief(self, real_action, **kwargs):
        """Belief update NOT done here; see _update_belief"""
        cur_true_state = self.cur_state  # oopomdp_state
        next_true_state = self.transition_func(cur_true_state, real_action)
        real_observation = self.observation_func(next_true_state, real_action)

        if real_action != "detect":
            self._gridworld.move_robot(*Environment.ACTIONS[real_action])
        else:
            self._num_detect_actions += 1
        self.cur_state = next_true_state
        self._update_belief(real_action, real_observation, **kwargs)
        
        # Environment computes the reward based on true agent's states.
        reward = self._reward_func(cur_true_state, real_action, next_true_state)
        return reward, real_observation


    def _update_belief(self, real_action, real_observation, **kwargs):
        def robot_state_transition_func(robot_state, real_action, real_observation):
            # Returns robot_state after the transition
            robot_pose = robot_state['pose']
            if real_action != "detect":
                action = Environment.ACTIONS[real_action]
                next_robot_pose = self._gridworld.if_move_by(robot_pose, action[0], action[1])
                next_observed_targets = tuple(robot_state['observed'])
            else:
                # The 'detect' action is applied to targets within the field of view,
                # received by the observation; they will be marked as observed.
                next_observed_targets = tuple(robot_state['observed'])
                z, c = real_observation
                for j in range(len(c)):
                    target_id = self._first_target_id + c[j]
                    if target_id not in next_observed_targets:
                        next_observed_targets += (target_id,)
                next_observed_targets = tuple(sorted(next_observed_targets))
                next_robot_pose = robot_pose
            next_robot_state = Maze2D_ObjectState('robot', {'pose': next_robot_pose,
                                                            'observed': next_observed_targets})
            return next_robot_state

        # print(self.cur_belief.get_distribution(self._robot_id).mpe())
            
        print("updating belief>>>>")
        # OOPOMDP belief and normal POMDP belief update are different
        if isinstance(self.cur_belief, OOPOMDP_BeliefState):
            self.cur_belief.update(real_action, real_observation,
                                   observation_model=self.observation_model,
                                   transition_model=self.transition_model,
                                   robot_id=self._robot_id,
                                   robot_state_transition_func=robot_state_transition_func)
        else:
            self.cur_belief.update(real_action, real_observation, self, **kwargs)
        print(">>>>")
        self._gridworld.update_belief(self.cur_belief, self._first_target_id)

        
    def is_in_goal_state(self):
        # Environment checks whether the agent has detected N objects, or has made
        # N detect attempts. Print out agent's MPE beliefs of the N target poses.
        if self._num_detect_actions >= len(self._gridworld.target_poses):
            return True
        if isinstance(self.cur_belief.distribution, RandomPlanner.DummyDistribution):  # dummy distribution
            return False
        mpe = self.cur_belief.sample(sampling_method='max')
        count = 0
        detection_score = 0
        observed = set({})
        
        current_observed_targets = self.cur_state.get_object_state(self._robot_id)['observed']
        print(current_observed_targets)
        for objid in self.cur_state.object_states:
            if self.cur_state.get_object_class(objid) == "target":
                target_observed = objid in current_observed_targets
                if target_observed:
                    count +=1
                    observed.add(objid)
                mpe_target_pose = mpe.get_object_attribute(objid, 'pose')
                true_target_pose = self.cur_state.get_object_attribute(objid, 'pose')
                if true_target_pose == mpe_target_pose:
                    detection_score += math.exp(-dist(true_target_pose, mpe_target_pose))
        return count == len(self._gridworld.target_poses)

    # For POMCP particle reinvigoration
    def add_transform(self, state):
        """Used for particle reinvigoration"""
        for objid in state.object_states:
            if state.get_object_class(objid) == "target":
                target_pose = state.get_object_attribute(objid, "pose")
                target_x = max(0, min(self._gridworld.width-1, target_pose[0] + random.randint(-1, 1)))
                target_y = max(0, min(self._gridworld.height-1, target_pose[1] + random.randint(-1, 1)))
                state.get_object_state(objid)["pose"] = (target_x, target_y)

class Experiment:

    def __init__(self, env, oopomdp, planner, render=True, max_episodes=100, pilot=False):
        """
        if 'pilot' is True, the user provides action at every episode.
        """
        self._env = env
        self._planner = planner
        self._oopomdp = oopomdp
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
                  and (self._pilot or self._num_iter < self._max_episodes):

                reward = None
                if self._pilot:
                    for event in pygame.event.get():
                        action = self._env.on_event(event)
                        if action is not None:
                            action, reward, observation = self._planner.execute_next_action(action)
                            # searched_action = self._planner.search(self._planner._history)
                            # print("POMCP thinks %s next" % str(searched_action))
                            self._env._gridworld.update_observation(observation)
                else:
                    start_time = time.time()
                    action, reward, observation = \
                        self._planner.plan_and_execute_next_action()  # the action is a control to the robot
                    total_time += time.time() - start_time
                    self._env._gridworld.update_observation(observation)

                if reward is not None:
                    self._discounted_sum_rewards += reward #((self._planner.gamma ** self._num_iter) * reward)
                    print("---------------------------------------------")
                    print("%d: Action: %s; Reward: %.3f; Cumulative Reward: %.3f; Observation: %s"
                          % (self._num_iter, str(action), reward, self._discounted_sum_rewards, str(observation)))
                    print("---------------------------------------------")
                    rewards.append(reward)
                    self._num_iter += 1

                    if self._oopomdp.is_in_goal_state():
                        self._env.on_loop()
                        self._env.on_render()
                        break

                self._env.on_loop()
                self._env.on_render()
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
    

DETECT_FREQ=0.2
def _rollout_policy(tree, actions):
    # if random.random() > DETECT_FREQ:
    #     return random.choice(actions)
    # else:
    #     return 'detect'
    return random.choice(actions)

def initialize_oopomdp_dummy(sensor_params, gridworld, robot_id, first_target_id, num_particles=1000):
    num_targets = len(gridworld.target_poses)
    # init true state
    init_robot_state = Maze2D_ObjectState("robot", {'pose': gridworld.robot_pose, 'observed': ()})
    init_object_states = {robot_id: init_robot_state}
    for i in range(num_targets):
        target_state = Maze2D_ObjectState("target", {'pose': gridworld.target_poses[i]})
        target_id = i + first_target_id
        init_object_states[target_id] = target_state    
    init_true_state = OOPOMDP_State(init_object_states)
    init_belief = BeliefState(RandomPlanner.DummyDistribution())
    sys.stdout.write("Initializing OOPOMDP...")    
    oopomdp = Maze2D_OOPOMDP(gridworld, sensor_params, init_belief, init_true_state, robot_id, first_target_id)
    sys.stdout.write("done\n")
    return oopomdp    
    
    
def initialize_oopomdp_particles(sensor_params, gridworld, robot_id, first_target_id, prior="RANDOM", num_particles=1000):
    num_targets = len(gridworld.target_poses)
    init_robot_state = Maze2D_ObjectState("robot", {'pose': gridworld.robot_pose, 'observed': ()})

    # init belief
    particles = []
    while len(particles) < num_particles:
        init_object_states = {robot_id: init_robot_state}
        
        # object states
        for i in range(num_targets):
            if prior == "RANDOM":
                target_state = Maze2D_ObjectState("target", {'pose': (random.randint(0,gridworld.width-1),
                                                                      random.randint(0,gridworld.height-1))})
            else:
                target_state = Maze2D_ObjectState("target", {'pose': gridworld.target_poses[i]})
            target_id = i + first_target_id
            init_object_states[target_id] = target_state

        oopomdp_state = OOPOMDP_State(init_object_states)
        particles.append(oopomdp_state)
    init_belief = BeliefState(POMCP_Particles(particles))

    # init true state
    init_object_states = {robot_id: init_robot_state}
    for i in range(num_targets):
        target_state = Maze2D_ObjectState("target", {'pose': gridworld.target_poses[i]})
        target_id = i + first_target_id
        init_object_states[target_id] = target_state    
    init_true_state = OOPOMDP_State(init_object_states)
    
    sys.stdout.write("Initializing OOPOMDP...")    
    oopomdp = Maze2D_OOPOMDP(gridworld, sensor_params, init_belief, init_true_state, robot_id, first_target_id)
    sys.stdout.write("done\n")        
    return oopomdp
    

def initialize_oopomdp_histogram(sensor_params, gridworld, robot_id, first_target_id, prior="RANDOM"):
    num_targets = len(gridworld.target_poses)
    belief_states = {}
    init_object_states = {}
    # deterministic prior for robot pose
    init_robot_state = Maze2D_ObjectState("robot", {'pose': gridworld.robot_pose, 'observed': ()})
    init_object_states[robot_id] = init_robot_state
    
    robot_belief_distribution = OOPOMCP_Histogram(robot_id, {init_robot_state:1.0})
    robot_belief_state = Maze2D_BeliefState("robot", gridworld, robot_belief_distribution)
    belief_states[robot_id] = robot_belief_state

    # groundtruth prior
    if prior == "GROUNDTRUTH":
        for i in range(num_targets):
            histogram = {}
            for x in range(gridworld.width):
                for y in range(gridworld.height):
                    target_state = Maze2D_ObjectState("target", {'pose': (x,y)})
                    if (x,y) == gridworld.target_poses[i]:
                        histogram[target_state] = 1.0
                    else:
                        histogram[target_state] = 0.0
                    # histogram[target_state] = 1.0 / (gridworld.width * gridworld.height)
            target_id = i + first_target_id
            target_belief_distribution = OOPOMCP_Histogram(target_id, histogram)
            belief_states[target_id] = Maze2D_BeliefState("target", gridworld, target_belief_distribution)

            target_pose = gridworld.target_poses[i]
            target_state = Maze2D_ObjectState("target", {'pose':target_pose})
            init_object_states[target_id] = target_state
    elif prior == "RANDOM":
        # random prior for targets
        for i in range(num_targets):
            histogram = {}
            total_prob = 0
            for x in range(gridworld.width):
                for y in range(gridworld.height):
                    target_state = Maze2D_ObjectState("target", {'pose': (x,y)})
                    histogram[target_state] = random.random()#1.0 / (gridworld.width * gridworld.height)
                    total_prob += histogram[target_state]
            # normalize
            for state in histogram:
                histogram[state] /= total_prob
            target_id = i + first_target_id
            target_belief_distribution = OOPOMCP_Histogram(target_id, histogram)
            belief_states[target_id] = Maze2D_BeliefState("target", gridworld, target_belief_distribution)

            target_pose = gridworld.target_poses[i]
            target_state = Maze2D_ObjectState("target", {'pose':target_pose})
            init_object_states[target_id] = target_state
        
    init_belief = OOPOMDP_BeliefState(belief_states)
    init_true_state = OOPOMDP_State(init_object_states)

    # Building OOPOMDP
    sys.stdout.write("Initializing OOPOMDP...")    
    oopomdp = Maze2D_OOPOMDP(gridworld, sensor_params, init_belief, init_true_state, robot_id, first_target_id)
    sys.stdout.write("done\n")        
    return oopomdp

world0 = \
"""
R...
....
T..T
"""

def unittest():
    sensor_params = {
        'max_range':5,
        'min_range':1,
        'view_angles': math.pi/2,
        'sigma_dist': 0.01,
        'sigma_bearing': 0.01}
    gridworld = MultiTargetGridWorld(world0)
    env = MultiTargetEnvironment(gridworld,
                                 sensor_params,
                                 res=30, fps=10, controllable=False)
    # env.on_execute()
    robot_id = 0
    first_target_id = 1

    # # Random
    # oopomdp = initialize_oopomdp_dummy(sensor_params, gridworld, robot_id, first_target_id)
    # planner = RandomPlanner(oopomdp)
    
    # OOPOMCP
    oopomdp = initialize_oopomdp_histogram(sensor_params, gridworld, robot_id, first_target_id, prior="RANDOM")
    sys.stdout.write("Initializing Planner: OOPOMCP...")
    planner = OOPOMCP(oopomdp, max_depth=100, max_time=1., gamma=0.8,
                      rollout_policy=_rollout_policy, exploration_const=10)
    sys.stdout.write("done\n")

    # # POMCP; belief won't be visualized
    # num_particles = 5000
    # oopomdp = initialize_oopomdp_particles(sensor_params, gridworld, robot_id, first_target_id, prior="RANDOM",
    #                                        num_particles=num_particles)
    # sys.stdout.write("Initializing Planner: OOPOMCP...")
    # planner = POMCP(oopomdp, max_depth=30, max_time=3., gamma=0.9, num_particles=num_particles,
    #                 rollout_policy=_rollout_policy, exploration_const=0.2,
    #                 observation_based_resampling=False)
    # sys.stdout.write("done\n")   
    
    experiment = Experiment(env, oopomdp, planner, pilot=False, max_episodes=1000)
    print("Running experiment")
    experiment.run()    

if __name__ == '__main__':
    unittest()
        
        
