import copy
import random

from simple_rl.agents import QLearningAgent
from simple_rl.agents import RandomAgent
from simple_rl.mdp import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.tasks.trench.TrenchOOMDPState import TrenchOOMDPState


class TrenchOOMDP(OOMDP):
    ''' Class for a Trench OO-MDP '''

    # Static constants.
    ACTIONS = ["forward", "rotate_right", "rotate_left", "pickup", "place"]
    ATTRIBUTES = ["x", "y", "dx", "dy", "has_block", "dest_x", "dest_y"]
    CLASSES = ["agent", "block", "lava"]

    def __init__(self, width, height, agent, blocks, lavas, gamma=0.99, slip_prob=0.0, name="trench"):
        self.height = height
        self.width = width
        self.name = name

        agent_obj = OOMDPObject(attributes=agent, name="agent")
        block_objs = self._make_oomdp_objs_from_list_of_dict(blocks, "block")
        lava_objs = self._make_oomdp_objs_from_list_of_dict(lavas, "lava")

        init_state = self._create_state(agent_obj, block_objs, lava_objs)
        OOMDP.__init__(self, TrenchOOMDP.ACTIONS, self._trench_transition_func, self._trench_reward_func, init_state=init_state, gamma=gamma)
        self.slip_prob = slip_prob

    def _create_state(self, agent_oo_obj, blocks, lavas):
        '''
        Args:
            agent_oo_obj (OOMDPObjects)
            blocks (list of OOMDPObject)
            lavas (list of OOMDPObject)

        Returns:
            (OOMDP State)
        '''

        objects = {c : [] for c in TrenchOOMDP.CLASSES}

        objects["agent"].append(agent_oo_obj)

        # Make walls.
        for b in blocks:
            objects["block"].append(b)

        # Make passengers.
        for l in lavas:
            objects["lava"].append(l)

        return TrenchOOMDPState(objects)

    def _trench_reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''

        if self._is_goal_state_action(state, action):
            return 10.0
        elif self._is_lava_state_action(state, action):
            return -1.0
        else:
            return 0

    def _is_goal_state_action(self, state, action):
        if action == "forward":
            agent = state.get_first_obj_of_class("agent")

            next_x = agent.get_attribute("x") + agent.get_attribute("dx")
            next_y = agent.get_attribute("y") + agent.get_attribute("dy")
            if next_x == agent.get_attribute("dest_x") and next_y == agent.get_attribute("dest_y"):
                return True
            else:
                return False
        return False

    def _is_lava_state_action(self, state, action):
        if action == "forward":
            agent = state.get_first_obj_of_class("agent")

            next_x = agent.get_attribute("x") + agent.get_attribute("dx")
            next_y = agent.get_attribute("y") + agent.get_attribute("dy")
            for l in state.get_objects_of_class("lava"):
                if next_x == l.get_attribute("x") and next_y == l.get_attribute("y"):
                    return True
        return False

    def _is_goal_state(self, state):
        agent = state.get_first_obj_of_class("agent")

        next_x = agent.get_attribute("x")
        next_y = agent.get_attribute("y")
        if next_x == agent.get_attribute("dest_x") and next_y == agent.get_attribute("dest_y"):
            return True
        else:
            return False

    def _is_lava_state(self, state):
        agent = state.get_first_obj_of_class("agent")

        next_x = agent.get_attribute("x")
        next_y = agent.get_attribute("y")
        for l in state.get_objects_of_class("lava"):
            if next_x == l.get_attribute("x") and next_y == l.get_attribute("y"):
                return True
        return False

    def _trench_transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state

        r = random.random()

        if self.slip_prob > r:
            # Flip dir.
            if action == "forward":
                action = random.choice(["rotate_left", "rotate_right", "place"])
            elif action == "rotate_left":
                action = random.choice(["forward", "rotate_right", "place"])
            elif action == "rotate_right":
                action = random.choice(["forward", "rotate_left", "place"])

        forward_state_in_bounds = self._forward_state_in_bounds(state)
        is_forward_loc_block = self._is_forward_loc_block(state)
        if action == "forward" and forward_state_in_bounds and not is_forward_loc_block:
            next_state = self.move_agent_forward(state)
        elif action == "rotate_left":
            next_state = self.rotate_agent_left(state)
        elif action == "rotate_right":
            next_state = self.rotate_agent_right(state)
        elif action == "pickup" and is_forward_loc_block:
            next_state = self.agent_pickup(state)
        elif action == "place" and state.get_first_obj_of_class("agent").get_attribute("has_block") and \
                forward_state_in_bounds and not is_forward_loc_block:
            next_state = self.agent_place(state)
        else:
            next_state = state

        if self._is_terminal_state(next_state):
            next_state.set_terminal(True)

        # All OOMDP states must be updated.
        next_state.update()

        return next_state

    def _is_terminal_state(self, state):
        return self._is_goal_state(state) or self._is_lava_state(state)

    def _is_forward_loc_block(self, state):
        agent = state.get_first_obj_of_class("agent")

        next_x = agent.get_attribute("x") + agent.get_attribute("dx")
        next_y = agent.get_attribute("y") + agent.get_attribute("dy")

        for b in state.get_objects_of_class("block"):
            if next_x == b["x"] and next_y == b["y"]:
                return True
        return False

    def _forward_state_in_bounds(self, state):
        agent = state.get_first_obj_of_class("agent")

        next_x = agent.get_attribute("x") + agent.get_attribute("dx")
        next_y = agent.get_attribute("y") + agent.get_attribute("dy")

        x_check = 1 <= next_x <= self.width
        y_check = 1 <= next_y <= self.height
        return x_check and y_check

    def move_agent_forward(self, state):
        next_state = copy.deepcopy(state)

        agent_att = next_state.get_first_obj_of_class("agent").get_attributes()
        agent_att["x"] += agent_att["dx"]
        agent_att["y"] += agent_att["dy"]

        return next_state

    def rotate_agent_left(self, state):
        next_state = copy.deepcopy(state)

        agent_att = next_state.get_first_obj_of_class("agent").get_attributes()
        curr_dir = (agent_att["dx"], agent_att["dy"])

        dir_updates = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        agent_att["dx"], agent_att["dy"] = dir_updates[(dir_updates.index(curr_dir) + 1) % len(dir_updates)]
        return next_state

    def rotate_agent_right(self, state):
        next_state = copy.deepcopy(state)

        agent_att = next_state.get_first_obj_of_class("agent").get_attributes()
        curr_dir = (agent_att["dx"], agent_att["dy"])

        dir_updates = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        agent_att["dx"], agent_att["dy"] = dir_updates[(dir_updates.index(curr_dir) + 1) % len(dir_updates)]
        return next_state

    def agent_pickup(self, state):
        next_state = copy.deepcopy(state)

        agent = next_state.get_first_obj_of_class("agent")
        next_x = agent.get_attribute("x") + agent.get_attribute("dx")
        next_y = agent.get_attribute("y") + agent.get_attribute("dy")

        agent.set_attribute("has_block", 1)
        block_remove = 0
        for b in next_state.get_objects_of_class("block"):
            if next_x == b["x"] and next_y == b["y"]:
                break
            block_remove += 1
        next_state.get_objects_of_class("block").pop(block_remove)
        return next_state

    def agent_place(self, state):
        next_state = copy.deepcopy(state)

        agent = next_state.get_first_obj_of_class("agent")

        agent.set_attribute("has_block", 0)
        next_x = agent.get_attribute("x") + agent.get_attribute("dx")
        next_y = agent.get_attribute("y") + agent.get_attribute("dy")

        if self._is_lava_state_action(next_state, "forward"):
            lava_remove = 0
            for l in next_state.get_objects_of_class("lava"):
                if next_x == l.get_attribute("x") and next_y == l.get_attribute("y"):
                    break
                lava_remove += 1

            next_state.get_objects_of_class("lava").pop(lava_remove)
        else:
            new_block = {"x": next_x, "y": next_y}
            new_block_obj = self._make_oomdp_objs_from_list_of_dict([new_block], "block")
            next_state.get_objects_of_class("block").append(new_block_obj[0])

        return next_state

    def __str__(self):
        prefix = self.name
        return prefix + "_h-" + str(self.height) + "_w-" + str(self.width)


def main():
    # Setup MDP, Agents.
    size = 5
    agent = {"x": 1, "y": 1, "dx": 1, "dy": 0, "dest_x": size, "dest_y": size, "has_block": 0}
    blocks = [{"x": size, "y": 1}]
    lavas = [{"x": x, "y": y} for x, y in map(lambda z: (z + 1, (size + 1) / 2), range(size))]

    mdp = TrenchOOMDP(size, size, agent, blocks, lavas)
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=30, episodes=250, steps=250)

if __name__ == "__main__":
    main()
