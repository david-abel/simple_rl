# Python imports.
import Queue
import matplotlib.pyplot as plt
import math as m
import os

# Non-standard imports.
from simple_rl.tasks import ChainMDP, GridWorldMDP
try:
	import networkx as nx
except:
	print "Error: package networkx is required to use the visualize_mdp script."
	quit()

colors = [[240, 163, 255], [113, 113, 198],[197, 193, 170],\
            [113, 198, 113],[85, 85, 85], [198, 113, 113],\
            [142, 56, 142], [125, 158, 192],[184, 221, 255],\
            [153, 63, 0], [142, 142, 56], [56, 142, 142]]
colors = [[c[0]/255.0, c[1]/255.0, c[2]/255.0] for c in colors]

def visualize_mdp(mdp, state_abstr=lambda x: x, file_name="mdp.png"):
	'''
	Args:
		mdp (MDP)

	Summary:
		Creates a graph visual of the given MDP.
	'''
	graph_mdp, init_node = _make_graph_mdp(mdp, state_abstr=state_abstr)

	print "Num States:", len(graph_mdp.nodes())
	# print "Edges:", len(graph_mdp.edges())

	_make_graph_visual(graph_mdp, init_node, file_name=file_name)

def _make_graph_mdp(mdp, state_abstr, action_trials=5):
	'''
	Args:
		mdp (MDP)
		state_abstr (func: State --> int)
		action_trials (int): number of times to try an action to determine connectivity.

	Returns:
		(networkx.DiGraph, int): the latter represents the initial state node.

	Summary:
		Converts a simple_rl.MDP instance into a graph.
	'''

	# Get MDP components.
	state_queue = Queue.Queue()
	A = mdp.get_actions()
	R = mdp.get_reward_func()
	T = mdp.get_transition_func()
	s_init = mdp.get_init_state()

	# Data structures for tracking states.
	mdp_graph = nx.DiGraph() # Directed Graph.
	state_queue.put(s_init)
	visited = [s_init]

	while not state_queue.empty():

		state = state_queue.get()
		mdp_graph.add_node(hash(state_abstr(state)))

		if state.is_terminal():
			continue

		for action in A:
			# Try multiple times to get all "reasonable" transitions.
			for i in range(action_trials):

				# Adding a node that's already in there won't do anything.
				next_state = T(state, action)
				reward = R(state, action)

				if not ((state == next_state) == (hash(state) == hash(next_state))):
					print state, next_state
					print hash(state_abstr(state)), hash(state_abstr(next_state))
					print

				# Add next state and directed edge from @state to @next_state
				if next_state not in visited:
					visited.append(next_state)
					state_queue.put(next_state)
					if hash(state_abstr(next_state)) not in mdp_graph.nodes():
						mdp_graph.add_node(hash(state_abstr(next_state)))


				mdp_graph.add_edge(hash(state_abstr(state)), hash(state_abstr(next_state)), r=reward, a=action)

	return mdp_graph, hash(state_abstr(s_init))

def _color_starts_goals(graph_mdp, init_node, pos, reward_threshold=0.5):
	'''
		Args:
			graph_mdp (networkx.Graph)
			pos (networkx.Layout)
			reward_threshold (float): colorize all nodes with a transition with reward >= than this value.
	'''
	goal_edges = []
	goal_nodes = []

	for e in graph_mdp.edges(data=True):
		if e[2]["r"] >= reward_threshold:
			goal_nodes.append(e[1])
			goal_edges.append((e[0], e[1]))

	# Colorize goal nodes.
	nx.draw_networkx_nodes(graph_mdp, pos, nodelist=goal_nodes, node_size=300, node_color=colors[3])

	# Colorize start nodes.
	nx.draw_networkx_nodes(graph_mdp, pos, nodelist=[init_node], node_size=300, node_color="white")

	# Colorize goal edges.
	goal_edge_colors = len(goal_edges) * [colors[3]]
	nx.draw_networkx_edges(graph_mdp, pos, edgelist=goal_edges, width=1, edge_color=goal_edge_colors)


def _make_graph_visual(graph_mdp, init_node, file_name):
	'''
	Args:
		graph_mdp (networkx.Graph)
		init_node (int)
		file_name (str)

	Summary:
		Draws the given graph mdp.
	'''

	# Set layout.
	pos = nx.spring_layout(graph_mdp, iterations=500, k=0.35/m.sqrt(len(graph_mdp.nodes())))
	# pos = nx.fruchterman_reingold_layout(graph_mdp, iterations=2000, k=0.3/m.sqrt(len(graph_mdp.nodes())))

	# Nodes, edges, labels.
	nx.draw_networkx_nodes(graph_mdp, pos, node_list=graph_mdp.nodes(), node_size=300, node_color=colors[0])
	edge_colors = len(graph_mdp.edges()) * [colors[1]]
	nx.draw_networkx_edges(graph_mdp, pos, edgelist=graph_mdp.edges(), width=1, edge_color=edge_colors, style='solid')
	# nx.draw_networkx_labels(graph_mdp, pos, font_size=12, font_family='sans-serif')

	# Color the goals.
	_color_starts_goals(graph_mdp, init_node, pos)

 	# Display
 	try:
 		os.remove(file_name)
 	except:
 		pass
	plt.axis('off')
	plt.savefig(file_name)
	# plt.show()
	plt.cla() #clear

def main():
	mdp = GridWorldMDP(3, 5, (1, 1), [(3, 5)])

	vi = ValueIteration(mdp)
	# print "num states:", len(vi._compute_reachable_state_space())

	visualize_mdp(mdp)

if __name__ == "__main__":
	main()