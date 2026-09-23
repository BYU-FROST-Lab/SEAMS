#!/usr/bin/python3

# Original author:
# jplaced@unizar.es
# 2022, Universidad de Zaragoza

# Customized by:
# bab272@byu.edu
# 2026, Brigham Young University

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Include modules~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter

# from visualization_msgs.msg import MarkerArray
# from geometry_msgs.msg import Point

# from functions import createMarker

from shapely.geometry import Polygon
from shapely import affinity

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

import gtsam

debug = False

from common.fgHelpers import createSubGraph


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Main Class~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class WeightedPoseGraph:
    # Constructor
    def __init__(self, gtsamGraph:gtsam.NonlinearFactorGraph=gtsam.NonlinearFactorGraph(), 
                 gtsamValues:gtsam.Values=gtsam.Values(), 
                 criteria=['t_opt', 'e_opt'], 
                 keysToUse:list[int]=[]
                 ):
        '''
        criteria should be a list of any of the following 'd_opt', 't_opt', or 'e_opt'
        '''

        self.nx_graph = nx.Graph()
        self.criteria = criteria
        self._init_laplacian_storage()

        if keysToUse:
            gtsamGraph, gtsamValues = createSubGraph(gtsamGraph, gtsamValues, keysToUse)

        for k in gtsamValues.keys():
            self._add_node(k, gtsamValues)
        for i in range(gtsamGraph.size()):
            f = gtsamGraph.at(i)
            self._add_factor_edge(f)

    # ------------------------------------------------------------------
    # Incremental Laplacian (NetworkX rebuilds L from scratch every call)
    # ------------------------------------------------------------------

    def _init_laplacian_storage(self) -> None:
        self._n = 0
        self._cap = 0
        self._key_to_idx: dict[int, int] = {}
        self._idx_to_key: list[int] = []
        self._weight_attrs: list[str] = []
        if 'd_opt' in self.criteria:
            self._weight_attrs.append('weight_d')
        if 't_opt' in self.criteria:
            self._weight_attrs.append('weight_t')
        if 'e_opt' in self.criteria:
            self._weight_attrs.append('weight_e')
        if not self._weight_attrs:
            self._weight_attrs = ['weight']
        self._L: dict[str, np.ndarray] = {w: np.zeros((0, 0)) for w in self._weight_attrs}
        self._trace: dict[str, float] = {w: 0.0 for w in self._weight_attrs}

    def _grow_laplacians(self, min_cap: int) -> None:
        new_cap = max(16, self._cap * 2, min_cap)
        for w in self._weight_attrs:
            L_new = np.zeros((new_cap, new_cap))
            if self._n:
                L_new[:self._n, :self._n] = self._L[w][:self._n, :self._n]
            self._L[w] = L_new
        self._cap = new_cap

    def _laplacian_view(self, weight: str) -> np.ndarray:
        return self._L[weight][:self._n, :self._n]

    def _apply_edge_delta(self, k0: int, k1: int, data: dict, sign: float) -> None:
        """Rank-2 Laplacian update for edge (k0, k1): L += sign * (w on diag, -w off-diag)."""
        if k0 == k1:
            return
        i = self._key_to_idx[k0]
        j = self._key_to_idx[k1]
        for attr in self._weight_attrs:
            w = sign * float(data.get(attr, 0.0))
            if w == 0.0:
                continue
            L = self._L[attr]
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
            self._trace[attr] += 2.0 * w

    # ------------------------------------------------------------------
    # Private helpers for node/edge insertion
    # ------------------------------------------------------------------

    def _add_node(self, k: int, gtsamValues: gtsam.Values) -> None:
        """Insert node k if it is not already in the graph."""
        if self.nx_graph.has_node(k):
            return
        pose3 = gtsamValues.atPose3(k)
        p = [pose3.x(), pose3.y()]
        self.nx_graph.add_node(k, pose=p, theta=pose3.rotation().yaw())
        if self._n >= self._cap:
            self._grow_laplacians(self._n + 1)
        self._key_to_idx[k] = self._n
        self._idx_to_key.append(k)
        self._n += 1

    def _remove_node(self, k: int) -> None:
        """Remove node k and its incident edges, updating the Laplacian in place."""
        if not self.nx_graph.has_node(k):
            return
        if k in self._key_to_idx:
            for nbr, data in list(self.nx_graph[k].items()):
                self._apply_edge_delta(k, nbr, data, sign=-1.0)
            i = self._key_to_idx[k]
            last = self._n - 1
            if i != last:
                for attr in self._weight_attrs:
                    L = self._L[attr]
                    n = self._n
                    L[[i, last], :n] = L[[last, i], :n]
                    L[:n, [i, last]] = L[:n, [last, i]]
                last_key = self._idx_to_key[last]
                self._idx_to_key[i] = last_key
                self._key_to_idx[last_key] = i
            for attr in self._weight_attrs:
                L = self._L[attr]
                L[last, :self._n] = 0.0
                L[:self._n, last] = 0.0
            self._idx_to_key.pop()
            del self._key_to_idx[k]
            self._n -= 1
        self.nx_graph.remove_node(k)

    def _compute_edge_weights(self, info: np.ndarray) -> tuple[float, float, float]:
        """Return (weight_d, weight_t, weight_e) from a 6x6 information matrix."""
        eigv = np.linalg.eigvalsh(info)
        opt_cri_d = 0.0
        opt_cri_t = 0.0
        opt_cri_e = 0.0
        if 'd_opt' in self.criteria:
            eigv_pos = eigv[eigv > 1e-8]
            n = info.shape[0]
            opt_cri_d = float(np.exp(np.sum(np.log(eigv_pos)) / n)) if eigv_pos.size else 0.0
        if 't_opt' in self.criteria:
            opt_cri_t = float(np.trace(info))
        if 'e_opt' in self.criteria:
            opt_cri_e = float(max(eigv[0], 0.0))
        return opt_cri_d, opt_cri_t, opt_cri_e

    def _add_factor_edge(self, f) -> bool:
        """
        Add the edge for a 2-key BetweenFactorPose3 or LinearContainerFactor if
        both endpoints are already nodes in the graph and the edge is new.
        Returns True if an edge was added.
        """
        if not (isinstance(f, gtsam.BetweenFactorPose3) or isinstance(f, gtsam.LinearContainerFactor)):
            return False
        if len(f.keys()) != 2:
            return False
        keys = f.keys()
        k0, k1 = keys[0], keys[1]
        if not (self.nx_graph.has_node(k0) and self.nx_graph.has_node(k1)):
            return False
        if self.nx_graph.has_edge(k0, k1):
            return False

        # Paper 2602.17226 §III-B1 / Placed & Castellanos 2022: weight each
        # pose-graph edge by a scalar of the Fisher information it encodes.
        # T-opt uses the trace; E-opt uses the minimum eigenvalue.
        if isinstance(f, gtsam.LinearContainerFactor):
            info = f.toHessian().information()
        else:
            info = np.asarray(f.noiseModel().information(), dtype=float)
        opt_cri_d, opt_cri_t, opt_cri_e = self._compute_edge_weights(info)

        idx0 = gtsam.Symbol(k0).index()
        idx1 = gtsam.Symbol(k1).index()
        same_chr = gtsam.Symbol(k0).chr() == gtsam.Symbol(k1).chr()
        etype = 0 if (same_chr and abs(idx0 - idx1) == 1) else 1
        edge_data = dict(type=etype, information=info.flatten(),
                         weight_d=opt_cri_d, weight_t=opt_cri_t, weight_e=opt_cri_e)
        self.nx_graph.add_edge(k0, k1, **edge_data)
        self._apply_edge_delta(k0, k1, edge_data, sign=1.0)
        return True

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update(self,
               gtsamGraph: gtsam.NonlinearFactorGraph,
               gtsamValues: gtsam.Values,
               keys_to_keep: list[int]) -> None:
        """Incrementally sync this graph and its Laplacian to the supplied keys.

        Steps:
        1. Remove stale nodes (and rank-2-update the Laplacian for their edges).
        2. Add nodes present in keys_to_keep but missing from the graph
           (in list order, so prior keys are added before window keys).
        3. Scan the GTSAM factor graph and add any new 2-key edges, updating L.
        """
        keep_set = set(keys_to_keep)

        # Step 1: remove stale nodes (and their Laplacian contributions)
        stale = [k for k in list(self.nx_graph.nodes) if k not in keep_set]
        for k in stale:
            self._remove_node(k)

        # Step 2: add new nodes (preserving list order)
        for k in keys_to_keep:
            if not self.nx_graph.has_node(k) and gtsamValues.exists(k):
                self._add_node(k, gtsamValues)

        # Step 3: add missing edges
        for i in range(gtsamGraph.size()):
            f = gtsamGraph.at(i)
            if f is not None:
                self._add_factor_edge(f)

    def addEdge(self, id1, id2, t, I):
        A = [[I[0], I[1], I[2]],
             [I[1], I[3], I[4]],
             [I[2], I[4], I[5]]]
        eigv2, _ = np.linalg.eig(A)
        eigv = eigv2[eigv2 > 1e-8]
        n = np.size(A, 1)
        if self.criteria == 'd_opt':
            opt_cri = np.exp(np.sum(np.log(eigv)) / n)
        else:
            opt_cri = 0
            print("Error. Optimality criterion should be D-opt.")
        self.nx_graph.add_edge(id1, id2, type=t, information=I, weight=opt_cri)

    # Creates a graph from an existing one (which may be frozen)
    def copy_graph(self, frozenG):
        self.nx_graph.clear()
        self.nx_graph = nx.Graph(frozenG)
        self._rebuild_laplacians_from_nx()

    def _rebuild_laplacians_from_nx(self) -> None:
        """Recompute stored Laplacians from the current NetworkX graph."""
        self._init_laplacian_storage()
        nodes = list(self.nx_graph.nodes)
        if nodes:
            self._grow_laplacians(len(nodes))
        for k in nodes:
            self._key_to_idx[k] = self._n
            self._idx_to_key.append(k)
            self._n += 1
        for u, v, data in self.nx_graph.edges(data=True):
            self._apply_edge_delta(u, v, data, sign=1.0)

    # Computes the adjacency matrix of the graph (scipy sparse matrices)
    def compute_A(self, weight='weight_d'):
        return nx.adjacency_matrix(self.nx_graph, nodelist=None, weight=weight).toarray()

    # Return the incrementally maintained Laplacian (Degree - Adjacency)
    def compute_L(self, weight='weight_d'):
        if weight not in self._L:
            return nx.laplacian_matrix(self.nx_graph, weight=weight).toarray()
        return self._laplacian_view(weight).copy()

    def compute_average_degree(self, weight='weight_d') -> float:
        """Average weighted node degree: tr(L)/m = (1/m) sum_{k=2}^{m} λ_k."""
        n = self._n
        if n == 0:
            return 0.0
        if weight in self._trace:
            return float(self._trace[weight] / n)
        return float(np.trace(self.compute_L(weight)) / n)

    def compute_spectrum(self, weight='weight_d'):
        n = self._n
        if n == 0:
            return np.array([])
        return np.linalg.eigvalsh(np.ascontiguousarray(self._laplacian_view(weight)))

    def compute_reduced_spectrum(self, weight='weight_d'):
        return self.compute_spectrum(weight)[1:]

    def compute_max_eig(self, weight='weight_d'):
        spec = self.compute_spectrum(weight)
        return 0.0 if spec.size == 0 else float(np.max(spec))

    def compute_algcon(self, normalized=False, weight='weight_d'):
        return nx.algebraic_connectivity(self.nx_graph, weight=weight, normalized=normalized, tol=1e-08,
                                         method='tracemin')

    def compute_fiedler_value(self, weight='weight_d'):
        n = self._n
        if n < 2:
            return 0.0
        eigen_v = np.linalg.eigvalsh(np.ascontiguousarray(self._laplacian_view(weight)))
        return float(eigen_v[1])


    # Gets the edges that represent a loop closure (type 1 edges)
    # Returns an ordered list of loop closure edges (id1, id2) by its order of
    # appearance, that is, sorted by id2
    def get_LC_edges(self):
        edges = self.nx_graph.edges.data('type')
        edges_LC = []
        for (u, v, wt) in edges:
            edges_LC.append([u, v]) if wt == 1 else True

        return sorted(edges_LC, key=itemgetter(1))


    # Gets the nodes where there is a LC
    def get_LC_nodes(self):
        nodes = self.nx_graph.nodes.data('pose')
        edges = self.nx_graph.edges.data('type')
        idx_LC = []
        for (u, v, wt) in edges:
            if wt == 1:
                idx_LC.append(u) if u not in idx_LC else True
                idx_LC.append(v) if v not in idx_LC else True
        nodes_LC = [ith_node if ith_idx == ith_node[0] else 0 for ith_idx in idx_LC for ith_node in nodes]
        nodes_LC = list(filter(lambda num: num != 0, nodes_LC))

        return nodes_LC

    # Gets the max node id
    def get_max_node_id(self):
        nodes = self.nx_graph.nodes.data('pose')
        max_id = 0
        for (u, p) in nodes:
            if u > max_id:
                max_id = u

        return max_id

    # Gets the number of edges of the graph
    def get_num_edges(self):
        return nx.number_of_edges(self.nx_graph)

    # Gets the number of edges of the graph
    def get_num_nodes(self):
        return nx.number_of_nodes(self.nx_graph)


    # Draws the pose graph (i.e. trajectory)
    # Optional parameters:
    # - none, only plots the XY trajectory
    # - draw_LC_simple, also plots the starting point and the LC locations
    # - draw_LC_complex, also plots LC connections
    def plot_graph(self, label='Data', color='Blue', draw_LC_simple=False, draw_LC_complex=True):
        if not hasattr(self, "ax"):
            self.fig, self.ax = plt.subplots()
        # plt.figure()
        self.ax.clear()
        if self.nx_graph is not None:
            print("Plotting graph")
            nodes = self.nx_graph.nodes.data('pose')
            poses_x = [el[1][0] for el in nodes]
            poses_y = [el[1][1] for el in nodes]
            self.ax.plot(poses_x, poses_y, '-', label=label, alpha=1, color=color)
            self.ax.set_title('Trajectory')

            if draw_LC_simple and not draw_LC_complex:
                poses_LC = list(zip(*self.get_LC_nodes()))[1]
                poses_LC_x = [el[0] for el in poses_LC]
                poses_LC_y = [el[1] for el in poses_LC]
                self.ax.plot(poses_LC_x, poses_LC_y, '.', label='LC', color='black', alpha=1, markersize=2)
                self.ax.plot(poses_x[0], poses_y[0], '*', label='Start', color='black', alpha=1, markersize=10)

            elif draw_LC_complex:
                self.ax.plot(poses_x[0], poses_y[0], '*', label='Start', color='black', alpha=1, markersize=10)
                label_added = False
                edges_LC = self.get_LC_edges()
                for (u, v) in edges_LC:
                    poses_LC_x = []
                    poses_LC_y = []
                    poses_LC_x.append(self.nx_graph.nodes[u]['pose'][0])
                    poses_LC_x.append(self.nx_graph.nodes[v]['pose'][0])
                    poses_LC_y.append(self.nx_graph.nodes[u]['pose'][1])
                    poses_LC_y.append(self.nx_graph.nodes[v]['pose'][1])
                    if not label_added:
                        self.ax.plot(poses_LC_x, poses_LC_y, linestyle='-', color='orange', label='LC', alpha=1, marker='o',
                                 markeredgecolor='b', markersize=2)
                        label_added = True
                    else:
                        self.ax.plot(poses_LC_x, poses_LC_y, linestyle='-', color='orange', alpha=1, marker='o',
                                 markeredgecolor='b', markersize=2)

    