def move_agent(state, dx):
	'''
	Args:
		state (BlockDudeState)
		dx (int)
		dy (int)

	Returns:
		(BlockDudeState)
	'''
	ax, ay = state.get_agent_x(), state.get_agent_y()
	objects = state.get_objects()

	# Change agent orientation.
	if dx > 0:
		objects["agent"][0]["face_left"] = 0
		objects["agent"][0]["face_right"] = 1
	elif dx < 0:
		objects["agent"][0]["face_left"] = 1
		objects["agent"][0]["face_right"] = 0

	if state.is_solid_object_at(ax + dx, ay):
		# Something in the way.
		return
	else:
		# If there's nothing in the way, move agent.
		objects["agent"][0]["x"] += dx
		_fall(state, "agent", 0)

		if objects["agent"][0]["carrying"]:
			# Move carried block, too.
			objects["block"][state.get_carried_block()]["x"] += dx
			objects["block"][state.get_carried_block()]["y"] += objects["agent"][0]["y"]


def climb(state):
	'''
	Args:
		state (BlockDudeState)

	Summary:
		Executes the climb action.
	'''

	objects = state.get_objects()

	# If clearance, move agent.
	ax, ay = state.get_agent_x(), state.get_agent_y()
	dx = -1 if objects["agent"][0]["face_left"] else 1
	if state.is_solid_object_at(ax + dx, ay) and not state.is_solid_object_at(ax + dx, ay + 1):
		objects["agent"][0]["x"] += dx
		objects["agent"][0]["y"] += 1
		if objects["agent"][0]["carrying"]:
			# Move carried block, too.
			objects["block"][state.get_carried_block()]["x"] += dx
			objects["block"][state.get_carried_block()]["y"] += y

def pickup(state):
	'''
	Args:
		state (BlockDudeState)

	Summary:
		Executes the pickup action.
	'''

	objects = state.get_objects()

	if objects["agent"][0]["carrying"]:
		# Hands already full
		return

	# Otherwise, pickup adjacent block.
	ax, ay = state.get_agent_x(), state.get_agent_y()
	dx = -1 if objects["agent"][0]["face_left"] else 1
	pickup_block_index = state.get_block_index_at_loc(ax + dx, ay)
	if pickup_block_index is not None:
		objects["agent"][0]["carrying"] = 1
		objects["block"][pickup_block_index]["carried"] = 1

def drop(state):
	'''
	Args:
		state (BlockDudeState)

	Summary:
		Executes the drop action.
	'''

	objects = state.get_objects()


	if objects["agent"][0]["carrying"] == 0:
		# Agent's hands are already empty.
		return

	dx = -1 if objects["agent"][0]["face_left"] else 1
	if state.is_solid_object_at(ax + dx, ay + 1):
		# Clearance in correct dir.
		objects["agent"][0]["carrying"] = 0
		carried_block_index = state.get_carried_block()
		objects["block"][carried_block_index]["carried"] = 0
		objects["block"][carried_block_index]["x"] += dx

		# Let the block fall.
		_fall(state, "block", carried_block_index)

def _fall(state, obj_class, obj_index):
	'''
	Args:
		state (BlockDudeState)
		obj_class (str): Specifies an OOMDP class (like 'agent', 'wall').
		obj_index (int)
	'''
	objects = state.get_objects()
	
	# Get current object location and the ground below it.
	col_x = objects[obj_class][obj_index]["x"]
	ground_y = objects[obj_class][obj_index]["y"]

	# Find the ground.
	while not state.is_solid_object_at(col_x, ground_y) and ground_y != 0:
		ground_y -= 1

	objects[obj_class][obj_index]["y"] = ground_y + 1
