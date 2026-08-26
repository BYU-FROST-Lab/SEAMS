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

from common.FGVizHelpers import createSubGraph


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

        if keysToUse:
            gtsamGraph, gtsamValues = createSubGraph(gtsamGraph, gtsamValues, keysToUse)

        for k in gtsamValues.keys():
            pose3 = gtsamValues.atPose3(k)
            p = [pose3.x(), pose3.y()]
            self.nx_graph.add_node(k, pose=p, theta=pose3.rotation().yaw()) # TODO: Make this 6D
        for i in range(gtsamGraph.size()):
            f = gtsamGraph.at(i)
            if not (isinstance(f, gtsam.BetweenFactorPose3) or isinstance(f, gtsam.LinearContainerFactor)):
                continue
            keys = f.keys()
            edge = (keys[0], keys[1])

            # Paper 2602.17226 §III-B1 / Placed & Castellanos 2022: weight each
            # pose-graph edge by a scalar of the Fisher information it encodes.
            # T-opt uses the trace; E-opt uses the minimum eigenvalue.
            if isinstance(f, gtsam.LinearContainerFactor):
                info = f.toHessian().information()
            else:
                info = np.asarray(f.noiseModel().information(), dtype=float)
            eigv = np.linalg.eigvalsh(info)
            opt_cri_d = 0.0
            opt_cri_t = 0.0
            opt_cri_e = 0.0
            if 'd_opt' in criteria:
                eigv_pos = eigv[eigv > 1e-8]
                n = info.shape[0]
                opt_cri_d = float(np.exp(np.sum(np.log(eigv_pos)) / n)) if eigv_pos.size else 0.0
            if 't_opt' in criteria:
                opt_cri_t = float(np.trace(info))
                # opt_cri_t = 1.0
            if 'e_opt' in criteria:
                opt_cri_e = float(max(eigv[0], 0.0))

            idx0 = gtsam.Symbol(keys[0]).index()
            idx1 = gtsam.Symbol(keys[1]).index()
            same_chr = gtsam.Symbol(keys[0]).chr() == gtsam.Symbol(keys[1]).chr()
            etype = 0 if (same_chr and abs(idx0 - idx1) == 1) else 1
            self.nx_graph.add_edge(*edge, type=etype, information=info.flatten(),
                                    weight_d=opt_cri_d, weight_t=opt_cri_t, weight_e=opt_cri_e)

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

    # Computes the adjacency matrix of the graph (scipy sparse matrices)
    def compute_A(self, weight='weight_d'):
        return nx.adjacency_matrix(self.nx_graph, nodelist=None, weight=weight).toarray()

    # Compute the Laplacian matrix of the graph, Laplacian = Degree - Adjacency
    def compute_L(self, weight='weight_d'):
        return nx.laplacian_matrix(self.nx_graph, weight=weight).toarray()

    def compute_average_degree(self, weight='weight_d') -> float:
        """Average weighted node degree: tr(L)/m = (1/m) sum_{k=2}^{m} λ_k."""
        n = self.nx_graph.number_of_nodes()
        if n == 0:
            return 0.0
        # return float(sum(dict(self.nx_graph.degree(weight=weight)).values()) / n)
        return float(np.trace(self.compute_L(weight)) / n)

    # Computes the anchored Laplacian matrix along a random node
    def compute_anchored_L(self, weight='weight_d'):
        L = nx.laplacian_matrix(self.nx_graph, weight=weight)
        idx_to_drop = np.random.randint(0, np.shape(L)[1], 1)
        if debug:
            print('Laplacian anchored through index' + format(idx_to_drop))
        C = L.tocoo()
        keep = ~np.in1d(C.col, idx_to_drop)
        C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
        C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
        C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))

        keep = ~np.in1d(C.row, idx_to_drop)
        C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
        C.row -= idx_to_drop.searchsorted(C.row)  # decrement column indices
        C._shape = (C.shape[0] - len(idx_to_drop), C.shape[1])

        return C.tocsr()

    # Computes all eigenvalues of the Laplacian
    def compute_spectrum(self, weight='weight_d'):
        return nx.laplacian_spectrum(self.nx_graph, weight=weight)

    # Compute the non-zero eigenvalues of the Laplacian
    def compute_reduced_spectrum(self, weight='weight_d'):
        eigen_v = nx.laplacian_spectrum(self.nx_graph, weight=weight)
        eigen_v_reduced = eigen_v[1:]
        return eigen_v_reduced

    def compute_max_eig(self, weight='weight_d'):
        return np.max(nx.laplacian_spectrum(self.nx_graph, weight=weight))

    # Computes the algebraic connectivity of the graph
    def compute_algcon(self, normalized=False, weight='weight_d'):
        return nx.algebraic_connectivity(self.nx_graph, weight=weight, normalized=normalized, tol=1e-08,
                                         method='tracemin')
    
    # Compute the second eigenvalue (Fiedler value) of the Laplacian
    def compute_fiedler_value(self, weight='weight_d'):
        n = self.nx_graph.number_of_nodes()
        if n < 2:
            return 0.0
        eigen_v = nx.laplacian_spectrum(self.nx_graph, weight=weight)
        return float(eigen_v[1])

    # Computes the Fiedler vector of the graph
    def compute_fiedler_vector(self, weight='weight_d'):
        return nx.fiedler_vector(self.nx_graph, weight=weight, normalized=False, tol=1e-08, method='tracemin')

    # Gets the edges that represent a loop closure (type 1 edges)
    # Returns an ordered list of loop closure edges (id1, id2) by its order of
    # appearance, that is, sorted by id2
    def get_LC_edges(self):
        edges = self.nx_graph.edges.data('type')
        edges_LC = []
        for (u, v, wt) in edges:
            edges_LC.append([u, v]) if wt == 1 else True

        return sorted(edges_LC, key=itemgetter(1))

    # Gets the node number list of LC
    def get_LC_nodes_list(self):
        edges = self.nx_graph.edges.data('type')
        idx_LC = []
        for (u, v, wt) in edges:
            if wt == 1:
                idx_LC.append([u, v])  # if u not in idx_LC[0] else True
                # idx_LC.append(v) if v not in idx_LC[1] else True
        idx_LC.sort(key=lambda x: x[1])

        return idx_LC

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

    # Gets the neighbors to node "id" within a certain radius
    def find_neighbors(self, index, radius):
        nodes = self.nx_graph.nodes.data('pose')
        poses_xy = []

        # Do not find neighbors in recent nodes
        th = index - 20
        limit = max(th, 0)
        for i in range(0, int(limit)):
            poses_xy.append(nodes[i])

        # Find radius neighbors
        if len(poses_xy) > 0:
            n = min(5, len(poses_xy))
            neigh = NearestNeighbors(n_neighbors=n, radius=radius)
            neigh.fit(poses_xy)
            neighbors = neigh.radius_neighbors([nodes[index]], return_distance=True, sort_results=True)
            neighbors = neighbors[1][0]  # Only indices
        else:
            neighbors = []

        return neighbors

    def find_overlapping_neighbors(self, index, radius):

        nodes = self.nx_graph.nodes.data('pose')
        nodes_th = self.nx_graph.nodes.data('theta')
        poses_xy = []
        pose_current = nodes[index]
        theta_current = nodes_th[index]

        # Do not find neighbors in recent nodes
        th = index - 20
        limit = max(th, 0)
        for i in range(0, int(limit)):
            poses_xy.append(nodes[i])

        # Find radius neighbors
        if len(poses_xy) > 0:
            n = min(5, len(poses_xy))
            neigh = NearestNeighbors(n_neighbors=n, radius=radius)
            neigh.fit(poses_xy)
            neighbors = neigh.radius_neighbors([pose_current], return_distance=True, sort_results=True)
            neighbors = neighbors[1][0]  # Only indices
        else:
            neighbors = []

        neighbors_overlap = []
        if len(neighbors) > 0:
            laser_range = 6
            p1 = [(pose_current[0] + laser_range), pose_current[1]]
            p2 = [(pose_current[0] - laser_range), pose_current[1]]
            p3 = [pose_current[0], (pose_current[1] + laser_range)]
            p4 = [(pose_current[0] + laser_range*np.cos(0.785398)), (pose_current[1] + laser_range*np.sin(0.785398))]
            p5 = [(pose_current[0] - laser_range*np.sin(0.785398)), (pose_current[1] + laser_range*np.cos(0.785398))]
            polygon_tmp = Polygon([pose_current, p1, p4, p3, p5, p2])
            polygon1 = affinity.rotate(polygon_tmp, theta_current, origin=(pose_current[0], pose_current[1]),
                                       use_radians=True)
            # fig, ax = plt.subplots()
            # self.plot_polygon(ax, polygon1, facecolor='lightblue', edgecolor='lightblue')

            for i in neighbors:
                pose_neigh = nodes[i]
                theta_neigh = nodes_th[i]
                p1 = [(pose_neigh[0] + laser_range), pose_neigh[1]]
                p2 = [(pose_neigh[0] - laser_range), pose_neigh[1]]
                p3 = [pose_neigh[0], (pose_neigh[1] + laser_range)]
                p4 = [(pose_neigh[0] + laser_range*np.cos(0.785398)), (pose_neigh[1] + laser_range*np.sin(0.785398))]
                p5 = [(pose_neigh[0] - laser_range*np.sin(0.785398)), (pose_neigh[1] + laser_range*np.cos(0.785398))]
                polygon_tmp = Polygon([pose_neigh, p1, p4, p3, p5, p2])
                polygon2 = affinity.rotate(polygon_tmp, theta_neigh, origin=(pose_neigh[0], pose_neigh[1]),
                                           use_radians=True)
                # self.plot_polygon(ax, polygon2, facecolor='red', edgecolor='red', alpha=0.05)

                intersection = polygon1.intersection(polygon2)
                A = intersection.area

                if A > 0.15 * polygon1.area:
                    neighbors_overlap.append(i)

            # ax.set_xlim([-25, 25])
            # ax.set_ylim([-25, 25])
            # plt.show()
        return neighbors_overlap

    @staticmethod
    def plot_polygon(ax, poly, **kwargs):
        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

        patch = PathPatch(path, **kwargs)
        collection = PatchCollection([patch], **kwargs)

        ax.add_collection(collection, autolim=True)
        ax.autoscale_view()
        return collection

    # Draws the pose graph (i.e. trajectory)
    # Optional parameters:
    # - none, only plots the XY trajectory
    # - draw_LC_simple, also plots the starting point and the LC locations
    # - draw_LC_complex, also plots LC connections
    def plot_graph(self, label='Data', color='Blue', draw_LC_simple=False, draw_LC_complex=True):
        plt.figure()
        if self.nx_graph is not None:
            print("Plotting graph")
            nodes = self.nx_graph.nodes.data('pose')
            poses_x = [el[1][0] for el in nodes]
            poses_y = [el[1][1] for el in nodes]
            plt.plot(poses_x, poses_y, '-', label=label, alpha=1, color=color)
            plt.suptitle('Trajectory')

            if draw_LC_simple and not draw_LC_complex:
                poses_LC = list(zip(*self.get_LC_nodes()))[1]
                poses_LC_x = [el[0] for el in poses_LC]
                poses_LC_y = [el[1] for el in poses_LC]
                plt.plot(poses_LC_x, poses_LC_y, '.', label='LC', color='black', alpha=1, markersize=2)
                plt.plot(poses_x[0], poses_y[0], '*', label='Start', color='black', alpha=1, markersize=10)

            elif draw_LC_complex:
                plt.plot(poses_x[0], poses_y[0], '*', label='Start', color='black', alpha=1, markersize=10)
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
                        plt.plot(poses_LC_x, poses_LC_y, linestyle='-', color='orange', label='LC', alpha=1, marker='o',
                                 markeredgecolor='b', markersize=2)
                        label_added = True
                    else:
                        plt.plot(poses_LC_x, poses_LC_y, linestyle='-', color='orange', alpha=1, marker='o',
                                 markeredgecolor='b', markersize=2)

    # def getGraphAsMarkerArray(self, global_frame: str = "map") -> MarkerArray:
    #     """
    #     Saves the graph as visualization_msgs/MarkerArray message for RViZ/ROS visualization
    #     """
    #     graph_marker = MarkerArray()
    #     graph_marker.markers.clear()
    #     id_markers = 1

    #     n = self.get_no_nodes()

    #     all_t = nx.get_node_attributes(self.nx_graph, 'pose')
    #     # Add vertices
    #     for i in range(1, n):
    #         vertex_marker = createMarker(mtype="sphere", frame=global_frame, ns="graph_ns", colors=[255, 255, 0],
    #                                      lifetime=15, alpha=1.0, scale=0.1)
    #         vertex_marker.id = id_markers

    #         if i in all_t:
    #             t = all_t[i]
    #             vertex_marker.pose.position.x = t[0]
    #             vertex_marker.pose.position.y = t[1]
    #             vertex_marker.pose.position.z = 0.0

    #             graph_marker.markers.append(vertex_marker)
    #             id_markers += 1

    #     # Add edges
    #     edge_marker = createMarker(mtype="lines", frame=global_frame, ns="graph_ns", colors=[0, 255, 0],
    #                                lifetime=15, alpha=1.0, scale=0.1)
    #     for (u, v, wt) in self.nx_graph.edges.data('type'):
    #         if u != 0 and v != 0:
    #             p = Point()
    #             # Edge's starting position
    #             t1 = all_t[int(u)]
    #             p.x = t1[0]
    #             p.y = t1[1]
    #             p.z = 0.0
    #             edge_marker.points.append(p)
    #             # Edge's ending position
    #             p = Point()
    #             t2 = all_t[int(v)]
    #             p.x = t2[0]
    #             p.y = t2[1]
    #             p.z = 0.0
    #             edge_marker.points.append(p)

    #     edge_marker.id = id_markers
    #     graph_marker.markers.append(edge_marker)

    #     return graph_marker
