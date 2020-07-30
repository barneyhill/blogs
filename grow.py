import numpy as np
from copy import deepcopy
from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.basis import Basis
from urllib.parse import quote

class PolyTree(object):

	def __init__(self, max_depth=5, min_samples_leaf=20, order=3, basis='tensor-grid', search='exhaustive', samples=10, logging=False):
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.order = order
		self.basis = basis
		self.tree = None
		self.search = search
		self.samples = samples
		self.logging = logging
		self.log = []
	
	def fit(self, X, y):
		"""
		Fits the tree to the provided data

		:param numpy.ndarray X:
			Training input data
		:param numpy.ndarray y:
			Training output data
		"""

		def _prune(node):
			def _fit_polys(node):

				mse, poly = _fit_poly(node['data'][0], node['data'][1])
				node["loss"] = mse
				node["poly"] = poly

				is_left = node["children"]["left"] != None
				is_right = node["children"]["right"] != None

				if is_left:
					node["children"]["left"] = _fit_polys(node["children"]["left"])
					
				if is_right:
					node["children"]["right"] = _fit_polys(node["children"]["right"])

				if is_left and is_right:
					lower_loss = ( node["children"]["left"]["loss"] * node["children"]["left"]["n_samples"] +
								 node["children"]["right"]["loss"] * node["children"]["right"]["n_samples"] ) / (node["children"]["left"]["n_samples"] + node["children"]["right"]["n_samples"])
					if lower_loss > node["loss"]:
						node["children"]["left"] = None
						node["children"]["right"] = None

				return node

			return _fit_polys(node)

		def _fit_poly(X, y):

			N, d = X.shape
			myParameters = []

			for dimension in range(d):
				values = [X[i,dimension] for i in range(N)]
				values_min = min(values)
				values_max = max(values)

				if (values_min - values_max) ** 2 < 0.01:
					myParameters.append(Parameter(distribution='Uniform', lower=values_min-0.01, upper=values_max+0.01, order=self.order))
				else: 
					myParameters.append(Parameter(distribution='Uniform', lower=values_min, upper=values_max, order=self.order))
			myBasis = Basis('total-order')
			
			y = np.reshape(y, (y.shape[0], 1))

			poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'sample-points':X, 'sample-outputs':y})

			poly.set_model()

			mse = ((y-poly.get_polyfit(X))**2).mean()
			return mse, poly

		def _build_tree():

			global index_node_global			
			
			def _splitter(node):
				# Extract data
				X, y = node["data"]
				depth = node["depth"]
				N, d = X.shape

				# Find feature splits that might improve loss
				did_split = False
				loss_best = np.inf
				data_best = None
				j_feature_best = None
				threshold_best = None

				# Perform threshold split search only if node has not hit max depth
				if (depth >= 0) and (depth <= self.max_depth):

					for j_feature in range(d):

						threshold_search = np.linspace(np.min(X[:,j_feature]), np.max(X[:,j_feature]), num=self.samples)
						
						# Perform threshold split search on j_feature
						for threshold in np.sort(threshold_search):

							# Split data based on threshold
							(X_left, y_left), (X_right, y_right) = _split_data(j_feature, threshold, X, y)
							#print(j_feature, threshold, X_left, X_right)
							N_left, N_right = len(X_left), len(X_right)

							# Do not attempt to split if split conditions not satisfied
							if not (N_left >= self.min_samples_leaf and N_right >= self.min_samples_leaf):
								continue

							# Compute weight loss function
							loss_split = np.std(y) - (N_left*np.std(y_left) + N_right*np.std(y_right)) / N	

							# Update best parameters if loss is lower
							if loss_split < loss_best:
								did_split = True
								loss_best = loss_split
								data_best = [(X_left, y_left), (X_right, y_right)]
								j_feature_best = j_feature
								threshold_best = threshold
	
							last_threshold = threshold
				# Return the best result
				result = {"did_split": did_split,
						  "loss": loss_best,
						  "data": data_best,
						  "j_feature": j_feature_best,
						  "threshold": threshold_best,
						  "N": N}

				return result

			def _split_data(j_feature, threshold, X, y):
				idx_left = np.where(X[:, j_feature] <= threshold)[0]
				idx_right = np.delete(np.arange(0, len(X)), idx_left)
				assert len(idx_left) + len(idx_right) == len(X)
				return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])
					
			def _create_node(X, y, depth, container):

				node = {"name": "node",
						"index": container["index_node_global"],
						"loss": np.inf,
						"data": (X, y),
						"n_samples": len(X),
						"j_feature": None,
						"threshold": None,
						"children": {"left": None, "right": None},
						"depth": depth}
				container["index_node_global"] += 1

				return node

			def _split_traverse_node(node, container):

				result = _splitter(node)
				if not result["did_split"]:
					self.log.append({"event": "UP"})
					return

				node["j_feature"] = result["j_feature"]
				node["threshold"] = result["threshold"]

				#del node["data"]

				(X_left, y_left), (X_right, y_right) = result["data"]

				node["children"]["left"] = _create_node(X_left, y_left, node["depth"]+1, container)
				node["children"]["right"] = _create_node(X_right, y_right, node["depth"]+1, container)

				_split_traverse_node(node["children"]["left"], container)
				_split_traverse_node(node["children"]["right"], container)	


			container = {"index_node_global": 0}
			root = _create_node(X, y, 0, container)
			_split_traverse_node(root, container)

			return root

		self.tree = _build_tree()
		self.tree = _prune(self.tree)
	
	def predict(self, X):
		"""
		Evaluates the the polynomial tree approximation of the data.

		:param numpy.ndarray X:
			An ndarray with shape (number_of_observations, dimensions) at which the tree fit must be evaluated at.
		:return: **y**:
			A numpy.ndarray of shape (1, number_of_observations) corresponding to the polynomial approximation of the tree.
		"""
		assert self.tree is not None
		def _predict(node, x):
			no_children = node["children"]["left"] is None and \
						  node["children"]["right"] is None
			if no_children:
				y_pred_x = node["poly"].get_polyfit(np.array(x))[0]
				return y_pred_x
			else:
				if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
					return _predict(node["children"]["left"], x)
				else:  # x[j] > threshold
					return _predict(node["children"]["right"], x)
		y_pred = np.array([_predict(self.tree, np.array(x)) for x in X])
		return y_pred

	def get_graphviz(self, feature_names, file_name):
		"""
		Returns a url to the rendered graphviz representation of the tree.

		:param list feature_names:
			A list of the names of the features used in the training data
		"""
		from graphviz import Digraph
		g = Digraph('g', node_attr={'shape': 'record', 'height': '.1'})

		def build_graphviz_recurse(node, parent_node_index=0, parent_depth=0, edge_label=""):

			# Empty node
			if node is None:
				return

			# Create node
			node_index = node["index"]
			if node["children"]["left"] is None and node["children"]["right"] is None:
				threshold_str = ""
			else:
				threshold_str = "{} <= {:.3f}\\n".format(feature_names[node['j_feature']], node["threshold"])
			
			#indices = []
			#for i in range(len(feature_names)):
			#	indices.append("{} : {}\\n".format(feature_names[i], node["poly"].get_sobol_indices(1)[i,]))
			label_str = "{} n_samples = {}\\n loss = {:.6f}".format(threshold_str, node["n_samples"], node["loss"])

			# Create node
			nodeshape = "rectangle"
			bordercolor = "black"
			fillcolor = "white"
			fontcolor = "black"
			g.attr('node', label=label_str, shape=nodeshape)
			g.node('node{}'.format(node_index),
				   color=bordercolor, style="filled",
				   fillcolor=fillcolor, fontcolor=fontcolor)

			# Create edge
			if parent_depth > 0:
				g.edge('node{}'.format(parent_node_index),
					   'node{}'.format(node_index), label=edge_label)

			# Traverse child or append leaf value
			build_graphviz_recurse(node["children"]["left"],
								   parent_node_index=node_index,
								   parent_depth=parent_depth + 1,
								   edge_label="")
			build_graphviz_recurse(node["children"]["right"],
								   parent_node_index=node_index,
								   parent_depth=parent_depth + 1,
								   edge_label="")

		# Build graph
		build_graphviz_recurse(self.tree,
							   parent_node_index=0,
							   parent_depth=0,
							   edge_label="")

		try:
			g.render(view=True)
		except:
			file_name = file_name + ".txt"
			with open(file_name, "w") as file:
				file.write(str(g.source))
				print("GraphViz source file written to " + file_name + " and can be viewed using an online renderer. Alternatively you can install graphviz on your system to render locally")