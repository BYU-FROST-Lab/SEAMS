import numpy as np
import FGVizHelpers as fgtools
import gtsam
import matplotlib.pyplot as plt
import copy
import time
import networkx as nx
from typing import Any
import cvxpy as cp
import scipy as sp
from matplotlib.lines import Line2D

class MethodStruct:
    def __init__(self, method:str):
        self.glc_dense = False
        self.nfr_tree = False

        if method == "glc_dense":
            self.glc_dense = True
        elif method == "nfr_tree":
            self.nfr_tree = True
        else:
            raise ValueError("method must be either 'glc_dense' or 'nfr_tree'")


class GraphSparsification :
    def __init__(self, graph = gtsam.NonlinearFactorGraph(), initial_values = gtsam.Values(), linearizationType = "local", showPlots=False, ) -> None:
        self._graph = copy.deepcopy(graph)
        print("\nInitial Optimization")
        self._values = self.Optimize(graph, initial_values, printTime=True)
        print()

        if linearizationType == "local":
            self.doLocalLinearization = True
        else:
            self.doLocalLinearization = False

        # Visualization Parameters
        self.showPlot = showPlots


        # Other globals
        self.markovGraph = gtsam.NonlinearFactorGraph()
        self.markovValues = gtsam.Values()
        self.initial_graph = copy.deepcopy(self._graph)
        self.initial_values = copy.deepcopy(self._values)

    @property
    def graph(self):
        return self._graph
    
    @property
    def values(self):
        return self._values

    def Optimize(self, graph: gtsam.gtsam.NonlinearFactorGraph, values: gtsam.gtsam.Values, n=1, printTime=False):
        tmpvalues = copy.deepcopy(values)
        if printTime:
            print("Optimizing using LevenbergMarquardt...")
        for i in range(n):
            startTime = time.time()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, tmpvalues)
            tmpvalues = optimizer.optimize()
            if printTime:
                if n > 1:
                    string = f"i: {i} "
                else:
                    string = ""
                print(f"{string}Optimization time: {time.time() - startTime}")
        return tmpvalues

    def ExtractMarkovBlanket(self, keysOfInterest: list, ):
        """
        Returns subgraph containing all nodes connected to the nodes of interest and all factors connecting any of those nodes
        
        :param keysOfInterest: List of central nodes to find all connecting nodes to
        :type keysOfInterest: list

        :return markovGraph: Nonlinear Factor Graph for the Markov blanket to the keysOfInterest
        :return markovValues: Values for the Markov blanket to the keysOfInterest
        """

        for k in keysOfInterest:
            if not self._values.exists(k):
                raise Exception(f"Key \"{k}\" does not exist in the values object anymore. \nValid keys are {self._values.keys()}.")
            
        # Get subgraph with intra-factors only for keysOfInterest and factors between keysOfInterest and next pose
        self.markovGraph = gtsam.NonlinearFactorGraph()
        self.markovValues = gtsam.Values()

        set_keysOfInterest = set(keysOfInterest)
        neighbor_keys = set()
        for i in range(self._graph.size()):
            factor = self._graph.at(i)
            if factor is None:
                continue
            factor_keys = set(factor.keys())

            if factor_keys & set_keysOfInterest:
                neighbor_keys |= factor_keys
                self.markovGraph.push_back(factor)
        
        for k in neighbor_keys:
            self.markovValues.insert(k, self._values.atPose3(k))

        # # Gets sub graph with intra-node connections
        # self.markovGraph, self.markovValues = fgtools.createSubGraph(self._graph, self._values, list(keysOfInterest))

        if self.doLocalLinearization:
            self.markovValues = self.Optimize(self.markovGraph, self.markovValues)

        return self.markovGraph, self.markovValues

    @staticmethod
    def PlotHessian(graph, values=None, title="", ax=None):
        if isinstance(graph, gtsam.gtsam.NonlinearFactorGraph):
            if values is None:
                raise Exception("Must pass a valid values object in order to linearize the NonlinearFactorGraph")
            # TODO: Look at graph.linearizeToHessianFactor() instead of just linearize
            gfg = graph.linearize(values)
            hessian = gfg.hessian()[0]
        elif isinstance(graph, gtsam.gtsam.GaussianFactorGraph):
            hessian = graph.hessian()[0]
        else:
            raise Exception(f"Current graph type ({type(graph)}) is invalid. Graph must be of type gtsam.gtsam.NonlinearFactorGraph or gtsam.gtsam.GaussianFactorGraph.")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.spy(hessian, precision=1e-9, markersize=5)
        ax.set_title(title)

    def MarginalizeNodes(self, keys, marginalizationType:str="remove", method:str="glc_dense", printTime=False):
        """
        Marginalize out nodes to create sparser factor graph
        
        :param keys: List of keys for poses
        :param marginalizationType: "remove" or "keep". Type of removal. I.e. are the poses to be removed or kept?
        :type marginalizationType: str
        :param method: "glc_dense", "nfr_tree". Type of new factor. "glc_dense" returns a dense linear n-nary factor. "nfr_tree" returns a Chow-Liu Tree non-linear factor approximation of the dense n-nary shur-complement
        :type method: str
        :param printTime: Dev. Print optimization times.
        :type printTime: bool

        :return graph: sparse graph 
        :type graph: gtsam.gtsam.NonlinearFactorGraph
        :return values: sparse values
        :type values: gtsam.gtsam.Values
        """

        if not (marginalizationType == "remove" or marginalizationType == "keep"):
            raise ValueError("marginalizationType must be either 'remove' or 'keep'")
        sparseMethod = MethodStruct(method)
        
        if marginalizationType == "remove":
            if printTime:
                print("Extracting Markov Blanket...")
                startTime = time.time()
            self.ExtractMarkovBlanket(list(keys))
            if printTime:
                print(f"Markov Blanket time: {time.time() - startTime}")

            gfg = self.markovGraph.linearize(self.markovValues)
            _, marginalGraph = gfg.eliminatePartialSequential(keys)
            newGraphKeys = list(set(self._values.keys()) - set(keys))
        else:
            gfg = self._graph.linearize(self._values)
            marginalGraph = gfg.marginal(keys)
            newGraphKeys = list(copy.copy(keys))
        
        if self.showPlot:
            print(f"Plotting markov graph...", end="\r", flush=True)
            numRows = 2
            numCols = 3
            _, axes = plt.subplots(numRows, numCols, figsize=(numCols*5,numRows*5))
            

            if marginalizationType == "remove":
                fgtools.plot_graph(self.markovGraph, self.markovValues, ax=axes[0][0])
                fgtools.plot_points(keys, self.markovValues, color="tab:red", ax=axes[0][0])
                keys_to_keep = list(set(self.markovValues.keys()) - set(keys))
                fgtools.plot_points(keys_to_keep, self.markovValues, color="tab:blue", ax=axes[0][0])

                legendElements = [
                    Line2D([0],[0], marker='o', color='w', label="Nodes to keep", markerfacecolor='tab:blue'),
                    Line2D([0],[0], marker='o', color='w', label="Nodes to remove", markerfacecolor='red'),
                ]
                axes[0][0].legend(handles=legendElements, loc='upper right')
            else:
                fgtools.plot_graph(self._graph, self._values, axes[0][0], color_override="tab:gray")
                fgtools.plot_points(keys, self._values, color="tab:red", ax=axes[0][0])

            axes[0][0].set_title("Markov Blanket")      

            print(f"Plotting initial hessian...", end="\r", flush=True)
            GraphSparsification.PlotHessian(self.initial_graph, self.initial_values, "Initial graph", ax=axes[1][0])
            print(f"Plotting initial hessian...Done")

        

        # Create new subgraph containing everything but the nodes to be removed
        self._graph, self._values = fgtools.createSubGraph(self._graph, self._values, newGraphKeys)

        # Add in new dense factor
        for i in range(marginalGraph.size()):
            hf = marginalGraph.at(i) # hessian factor
            if sparseMethod.glc_dense:
                marginalFactor = gtsam.LinearContainerFactor(hf)
                self._graph.add(marginalFactor)
            elif sparseMethod.nfr_tree:
                factorsToAdd = GraphSparsification.TreeNFR(hf, self._values)
                for f in factorsToAdd:
                    self._graph.add(f)


        self._values = self.Optimize(self._graph, self._values, printTime=printTime)
        if self.showPlot:
            print(f"Plotting marginalized graph...", end="\r", flush=True)
            fgtools.plot_graph(self._graph, self._values, ax=axes[0][1])
            GraphSparsification.PlotHessian(self._graph, self._values, "After marginalization", ax=axes[1][1])

            axes[0][1].set_title("Marginalized Graph")
            plt.tight_layout()

            # Plot confidence ellipses of old graph
            print(f"Plotting initial graph and ellipses...", end="\r", flush=True)
            fgtools.plot_graph(self.initial_graph, self.initial_values, axes[0][2], color_override="tab:gray")

            # Plot original covariances to compare against.
            # For visualization purposes, only plot the poses that are still in the sparse graph.
            fgtools.plot_graph_covariances(self._values.keys(), self.initial_graph, self.initial_values, axes[0][2], edgecolor='tab:blue', linestyle="--", alpha=0.7)

            # Plot confidence ellipses of new graph
            print(f"Plotting marginalized graph and ellipses...", end="\r", flush=True)
            fgtools.plot_graph(self._graph, self._values, axes[0][2])
            fgtools.plot_graph_covariances(self._values.keys(), self._graph, self._values, axes[0][2], edgecolor='red')
            
            axes[0][2].set_title(r"Overlaid graphs with 3$\sigma$ covariance ellipse")

            legendElements = [Line2D([0],[0], color='tab:blue', linestyle="--", alpha=0.5, label=r"Original 3$\sigma$ Cov Ellipse"),
                              Line2D([0],[0], color='red', label=r"Marginalized 3$\sigma$ Cov Ellipse")]
            axes[0][2].legend(handles=legendElements, loc='upper right')
            
            # Last plot not being used
            axes[1][2].set_visible(False)
            plt.tight_layout()
            print(f"Plotting marginalized graph and ellipses...Done")

        return self._graph, self._values
    
    def PlotGraphDiff(self):
        numRows = 2
        numCols = 2
        _, axes = plt.subplots(numRows, numCols, figsize=(numCols*6,numRows*5))
        
        print(f"Plotting initial graph with overlaid marginal graph...", end="\r", flush=True)
        fgtools.plot_graph(self.initial_graph, self.initial_values, axes[0][0], color_override="tab:gray")
        fgtools.plot_graph(self._graph, self._values, axes[0][0], alpha=0.7)
        axes[0][0].set_title(r"Original Graph with Marginalized Graph Overlaid")

        print(f"Plotting initial hessian...                           ", end="\r", flush=True)
        time.sleep(2)
        GraphSparsification.PlotHessian(self.initial_graph, self.initial_values, "Initial graph", ax=axes[1][0])

        print(f"Plotting marginalized graph with covariances...", end="\r", flush=True)
        fgtools.plot_points(self._values.keys(), self.initial_values, color="tab:gray", ax=axes[0][1])
        fgtools.plot_graph_covariances(self._values.keys(), self.initial_graph, self.initial_values, axes[0][1], edgecolor='tab:gray', linestyle="--", alpha=0.5)
        fgtools.plot_graph(self._graph, self._values, ax=axes[0][1], alpha=0.7, addKeyLabels=True)
        fgtools.plot_graph_covariances(self._values.keys(), self._graph, self._values, axes[0][1], edgecolor='red', alpha=0.7)
        axes[0][1].set_title(r"Marginalized Graph with 3$\sigma$ Covariance Ellipses")

        legendElements = [
            Line2D([0],[0], marker='o', color='w', label="Orig Nodes", markerfacecolor='tab:gray'),
            Line2D([0],[0], color='tab:gray', linestyle="--", alpha=0.5, label=r"Orig 3$\sigma$ Cov"),
            Line2D([0],[0], marker='o', color='w', label="Marg Nodes", markerfacecolor='tab:blue'),
            Line2D([0],[0], color='red', label=r"Marg 3$\sigma$ Cov"),
            ]
        axes[0][1].legend(handles=legendElements, loc='center left', bbox_to_anchor=(1, 0.5))

        print(f"Plotting marginalized hessian...               ", end="\r", flush=True)
        GraphSparsification.PlotHessian(self._graph, self._values, "After marginalization", ax=axes[1][1])
        
        plt.tight_layout()
        print(f"Plotting marginalized hessian...Done")

    # dfs() is modified from https://www.geeksforgeeks.org/dsa/detect-cycle-undirected-graph/ to use dictionaries instead of lists.
    @staticmethod
    def dfs(v, adj: dict, visited, parent, level=0, debug:bool=False):
        
        # Mark the current node as visited
        visited[v] = True
        tabs = ""
        if debug:
            for _ in range(level):
                tabs += "  |\t"
            print(tabs + f"node: {v} with neighbors {adj[v]}, visited: {visited}")
        # Recur for all the vertices adjacent to this vertex
        for neighbor in adj[v]:
            if debug:
                print(tabs + f"looking at neighbor node: {neighbor}")
            # If an adjacent vertex is not visited, 
            # then recur for that adjacent
            if not neighbor in visited:
                if debug:
                    print(tabs + f"neighbor: {neighbor} has not be visited yet")
                if GraphSparsification.dfs(neighbor, adj, visited, v, level=level+1, debug=debug):
                    if debug:
                        print(tabs + f"at node {v}: returning True")
                    return True
                    
            # If an adjacent vertex is visited and is not
            # parent of current vertex,
            # then there exists a cycle in the graph.
            elif neighbor != parent:
                if debug:
                    print(tabs + f"neighbor: {neighbor} != parent: {parent}. Returning True")
                return True
        if debug:
            print(tabs + f"End of tree. No cycle")
        return False

    # isCycle() is modified from https://www.geeksforgeeks.org/dsa/detect-cycle-undirected-graph/ to use dictionaries instead of lists.
    @staticmethod
    def isCycle(r, adj: dict, debug:bool=False):
        """
        :param r: key for root node
        :param adj: adjacency matrix for graph. Keys are nodes. Values are connections from node.
        :type adj: dict

        :return bool: True if there is a cycle. False if there isn't.    
        """
        if len(adj.keys()) == 0:
            print(f"isCycle: adjacency keys are len 0 Returning")
            return False
        
        if not r in adj: # If the root node is not in the adjacency matrix, it can't be a cycle
            print(f"isCycle: node {r} is not in adj {adj}")
            return False
        
        # Mark all the vertices as not visited
        visited = {}
        if debug:
            print("adjacency list: ", adj)
            print("root node: ", r)
        if GraphSparsification.dfs(r, adj, visited, -1, debug=debug):
                return True

        return False
    
    @staticmethod
    def PlotChowLiuTree(values, chowLiuTree, possibleConnections, CLT_ax=None, PC_ax=None):
        
        if PC_ax is None and CLT_ax is None:
            _, ax = plt.subplots(1, 2, figsize=(10,5))
            PC_ax = ax[0]
            CLT_ax = ax[1]


        G = nx.Graph()
        fk = set()
        for c in possibleConnections:
            G.add_edge(c[0], c[1], weight=c[2])
            fk |= set(c[:2])
        
        pos: dict[Any, tuple] = {}
        for k in fk:
            p = values.atPose3(k)
            pos[k] = (p.x(), p.y())

        # edge_labels = nx.get_edge_attributes(G, "weight")
        edge_labels = {}
        edge_colors = ["black"] * G.number_of_edges()
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            edge_labels[(u, v)] = f"{d['weight']:.3f}"
            for clt in chowLiuTree:
                
                if u in clt and v in clt:
                    edge_colors[i] = "red"
        
        if PC_ax is not None:
            nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", edge_color=edge_colors, ax=PC_ax)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=PC_ax)

            PC_ax.axis("equal")
            PC_ax.set_title("Possible Weighted Edges")

        G.clear_edges()
        
        priorNodes = []
        betweenNodes = list(G.nodes)
        for c in chowLiuTree:
            if c[1] is None: # Unary factor
                print(f"Adding unary factor to {c[0]}")
                priorNodes.append(f"P{c[0]}")
                G.add_edge(c[0], f"P{c[0]}", weight=1.)
                pos[f"P{c[0]}"] = (pos[c[0]][0], pos[c[0]][1] + 15.)
            else:
                G.add_edge(c[0], c[1], weight=c[2])

        edge_colors = ["red"] * G.number_of_edges()

        if CLT_ax is not None:
            nx.draw(G, pos, with_labels=True, nodelist=betweenNodes, node_size=750, node_color="lightblue", edge_color=edge_colors, ax=CLT_ax)
            if priorNodes:
                nx.draw_networkx_nodes(G, pos, nodelist=priorNodes, node_size=200, node_color="tab:green")
            CLT_ax.axis("equal")
            CLT_ax.set_title("Maximum Spanning Tree")

    @staticmethod
    def GetDirAdjFromAdj(adj: dict[int, list], node, parent=None):
        
        if parent is None:
            adj = copy.deepcopy(adj)
        else:
            adj[node].remove(parent)

        for child in adj[node]:
            adj = GraphSparsification.GetDirAdjFromAdj(adj, child, node)

        return adj

    @staticmethod
    def ChowLiuTreeFromHessianFactor(factor, addUnaryToRoot:bool=False, rootKey=None, debug: bool=False):
        n = 6 # Dimension of poses. 3 for Pose2. 6 for Pose3.

        def AddEdgeToAdj(k: int, v: int, adj: dict[int,list]):
            if k in adj:
                tmp = adj[k]
                tmp.append(v)
                adj[k] = tmp
            else:
                adj[k] = [v]

        possibleConnections = []
        fk = list(factor.keys())
        if len(fk) == 1: # Unary Factor
            possibleConnections.append([fk[0], None, None])
            return possibleConnections, possibleConnections, fk[0]
        elif len(fk) == 2 and not addUnaryToRoot: # Binary Factor
            possibleConnections.append([fk[0], fk[1], None])
            return possibleConnections, possibleConnections, fk[0]


        
        # Extract Mutual information
        info = factor.information()
        info += np.eye(len(info))

        Sigma = fgtools.choleskyInverse(info)

        nodeWeights = dict.fromkeys(fk, 0)
        nodeInfo = dict.fromkeys(fk, 0)
        for i in range(len(fk) -1):
            nodeInfo[fk[i]] = info[i*n+3,i*n+3] + info[i*n+4,i*n+4] + info[i*n+5,i*n+5] # Sum of x, y, z info
            for j in range(i+1, len(fk)):
                E_ii = Sigma[i*n:i*n+n, i*n:i*n+n]
                E_ij = Sigma[i*n:i*n+n, j*n:j*n+n]
                E_ji = Sigma[j*n:j*n+n, i*n:i*n+n]
                E_jj = Sigma[j*n:j*n+n, j*n:j*n+n]
                MI = 0.5 * np.log10(np.linalg.det(E_ii) * np.linalg.det(E_jj) / np.linalg.det(np.block([[E_ii, E_ij], [E_ji, E_jj]])))
                possibleConnections.append([fk[i], fk[j], MI])

                nodeWeights[fk[i]] += MI
                nodeWeights[fk[j]] += MI
                
        if rootKey is None:
            # rootKey = max(nodeWeights, key=nodeWeights.get) # Get's the key associated to the largest value
            rootKey = max(nodeInfo, key=nodeInfo.get)
        # Sort connections based on weights
        possibleConnections.sort(reverse=True, key=lambda x: x[2])

        if debug:
            print("Possible Connections:")
            print(possibleConnections)

        # Calc Chow-Liu Tree using Kruskal's algorithm
        chowLiuTree = []
        adj: dict[int, list] = {}
        for edge in possibleConnections:
            if debug:
                print(f"\n\nedge: {edge}")
                print(f"adj: {adj}")
                print(f"weight: {edge[2]}")
            # Check if new edge creates a cycle. If not, add it.
            addEdge = True
            if (edge[0] in adj) and (edge[1] in adj):
                if debug:
                    print("Checking for cycle")
                tmpAdj = copy.deepcopy(adj)
                AddEdgeToAdj(edge[0], edge[1], tmpAdj)
                AddEdgeToAdj(edge[1], edge[0], tmpAdj)
                if GraphSparsification.isCycle(edge[0], tmpAdj, debug=debug):
                    if debug:
                        print("Is a cycle")
                    addEdge = False
                else:
                    if debug:
                        print("Not a cycle")
            else:
                if debug:
                    print("Both nodes are not in adjacency dict. Adding edge")

            if addEdge:
                chowLiuTree.append(edge)
                AddEdgeToAdj(edge[0], edge[1], adj)
                AddEdgeToAdj(edge[1], edge[0], adj)
        
        dirAdj = GraphSparsification.GetDirAdjFromAdj(adj, rootKey)
        if debug:
            print("Adj:", adj)
            print("rootKey: ", rootKey)
            print("dirAdj:", dirAdj)
        for edge in chowLiuTree:
            if debug:
                print("old_edge: ", edge)
            if not (edge[1] in dirAdj[edge[0]]): # if edge[0] doesn't not point to edge[1], then switch direction
                tmp = edge[0]
                edge[0] = edge[1]
                edge[1] = tmp
            if debug:
                print("new_edge: ", edge,"\n")

        if addUnaryToRoot:
            print(f"Adding unary to root {rootKey}")
            chowLiuTree.append([rootKey, None, None])

        if debug:
            print("ChowLiuTree Connections")
            print(chowLiuTree)
        return chowLiuTree, possibleConnections, rootKey
    
    @staticmethod
    def ChowLiuTreeFromGraph(graph, values, plotGraphs: bool=True, debug:bool=False):
        
        chowLiuTreeDict = {}
        gfg = graph.linearize(values)
        # Determine possible connections
        for i in range(gfg.size()):
            f = gfg.at(i)

            Omega = f.information()
            zero_tol = 1e-6
            eig = np.linalg.eig(Omega)
            cnt = np.sum(np.array(eig.eigenvalues > zero_tol))
            print(f"cnt: {cnt}, evals: {len(eig.eigenvalues)}") # TODO: Remove after testing
            if cnt < len(eig.eigenvalues): 
                addUnaryToRoot = False
            else:
                addUnaryToRoot = True
            
            chowLiuTree, possibleConnections, _ = GraphSparsification.ChowLiuTreeFromHessianFactor(f,addUnaryToRoot=addUnaryToRoot, debug=debug)
            chowLiuTreeDict[tuple(f.keys())] = chowLiuTree

            if len(chowLiuTree) == 1:
                # there is only one edge so continue
                continue

            if plotGraphs:
                _, ax = plt.subplots(1, 3, figsize=(15, 5))
                fgtools.plot_graph(graph, values, ax=ax[0])
                ax[0].set_title("Original Graph")

                fk = list(f.keys())
                fgtools.plot_points(fk, values, color="tab:red", ax=ax[0])
                
                GraphSparsification.PlotChowLiuTree(values, chowLiuTree, possibleConnections, ax[2], ax[1])
                plt.show()

        return chowLiuTreeDict

    @staticmethod
    def TreeNFR(hessianFactor, values):
        Omega = hessianFactor.information()

        if len(hessianFactor.keys()) == 1: # Unary Factor
            print(f"Adding unary factor to {hessianFactor.keys()[0]}")
            return [gtsam.PriorFactorPose3(hessianFactor.keys()[0], values.atPose3(hessianFactor.keys()[0]), gtsam.noiseModel.Gaussian.Information(Omega))]
        
        # Use eigenvalues to estimate nullity robustly (matrix_rank can overestimate on near-singular matrices)
        eigvals = np.linalg.eigvalsh(Omega)
        max_eig = np.max(np.abs(eigvals)) if eigvals.size else 0.0
        zero_tol = max(max_eig * 1e-9, 1e-9)
        nullity = int(np.sum(eigvals <= zero_tol))
        rank = Omega.shape[0] - nullity
        print(f"rank: {rank}, evals: {len(eigvals)}, nullity: {nullity}, keys: {hessianFactor.keys()}") # TODO: Remove after testing

        # Pure relative pose constraints -> nullity == d (6 for Pose3)
        # GPS adds absolute translation -> nullity == 3 (rotation gauge only)
        addUnaryToRoot = (nullity == 0)
        addGpsUnary = (nullity == 3)

        chowLiuTree, possibleConnections, rootKey = GraphSparsification.ChowLiuTreeFromHessianFactor(hessianFactor, addUnaryToRoot=addUnaryToRoot, debug=False)

        nodeToIdx = {}
        for i, k in enumerate(hessianFactor.keys()):
            nodeToIdx[k] = i

        d = 6 # dimension of pose
        n = len(hessianFactor.keys()) # number of nodes
        if addUnaryToRoot:
            A = np.zeros((d*n, d*n))
        elif addGpsUnary:
            A = np.zeros((d*(n-1) + 3, d*n))
        else:
            A = np.zeros((d*(n-1), d*n))

        for k, (key_i, key_j, _) in enumerate(chowLiuTree):

            if key_j is None: # Unary factor
                H_i = np.eye(d) # Check: I believe that the jacobian of a unary factor is just the identity
                A[k*d:(k+1)*d, nodeToIdx[key_i]*d:(nodeToIdx[key_i]+1)*d] = H_i
                continue

            p_i = values.atPose3(key_i)
            p_j = values.atPose3(key_j)

            H_i = np.zeros((d, d), order="F")
            H_j = np.zeros((d, d), order="F")
            p_i.between(p_j, H_i, H_j)

            A[k*d:(k+1)*d, nodeToIdx[key_i]*d:(nodeToIdx[key_i]+1)*d] = H_i
            A[k*d:(k+1)*d, nodeToIdx[key_j]*d:(nodeToIdx[key_j]+1)*d] = H_j

        if addGpsUnary:
            # Add a translation-only unary on the root to preserve GPS-constrained DOF
            if rootKey is None:
                rootKey = list(hessianFactor.keys())[0]
            rootIdx = nodeToIdx[rootKey]
            start = len(chowLiuTree) * d
            A[start:start+3, rootIdx*d+3:rootIdx*d+6] = np.eye(3)

        if not addUnaryToRoot: # Is is not full rank and we need to project it to lower space
            eigvals, eigvects = np.linalg.eigh(Omega)
            idx = np.argsort(-eigvals) # Arg sort from greatest to smallest
            eigvals = eigvals[idx][:len(eigvals)-nullity] # Sort and remove the last 'nullity' eigenvalues which should be zero
            eigvects = eigvects[:,idx][:,:len(eigvects)-nullity]
            Pi = eigvects.T
            Omega = Pi @ Omega @ Pi.T
            A = A @ Pi.T
        
        Sigma = np.linalg.solve(Omega, np.eye(len(Omega)))

        # if addUnaryToRoot: # A should be full rank
        #     if np.isclose(np.linalg.det(A), 0):
        #         raise ValueError("The det of A matrix is 0. Not invertible.")
            
        #     start = time.time()
        #     Sigma_hat = np.zeros_like(Omega)

        #     n = int(len(Sigma) / d)
        #     print("n: ", n)

        #     # Equations 10 and 11 from NFR paper
        #     firstPass = True
        #     for i in range(n):
        #         for j in range(n):
        #             for k in range(n):
        #                 Aij = A[i*d:i*d+d, j*d:j*d+d]
        #                 Yjk = Sigma[j*d:j*d+d, k*d:k*d+d]
        #                 Aik = A[i*d:i*d+d, k*d:k*d+d]
        #                 Sigma_hat[i*d:i*d+d, i*d:i*d+d] += Aij @ Yjk @ Aik.T
        #                 print(f"\nijk: {i}{j}{k},\n  Aij: {Aij}\n  Yjk: {Yjk}\n  Aik.T: {Aik.T}\n  Sigma_hat: {Sigma_hat[i:i+d, i:i+d]}")
        #     print(f"Sigma1 time: {time.time() - start}")
        
        Sigma_hat = A @ Sigma @ A.T
        # print("Sigma_hat evals: \n", np.linalg.eigvals(Sigma_hat))

        # Sigma_hat should be positive definite. If not, fall back to dense factor to avoid NaNs downstream.
        try:
            np.linalg.cholesky(Sigma_hat)
        except Exception as e:
            print("Sigma_hat is not positive definite, falling back to dense factor: ", e)
            return [gtsam.LinearContainerFactor(hessianFactor)]

        block_infos = []
        I_d = np.eye(d)
        for i in range(len(chowLiuTree)):
            block = Sigma_hat[i*d:i*d+d, i*d:i*d+d]
            block_infos.append(np.linalg.solve(block, I_d))
        gps_info = None
        if addGpsUnary:
            start = len(chowLiuTree) * d
            block3 = Sigma_hat[start:start+3, start:start+3]
            gps_info = np.linalg.solve(block3, np.eye(3))
        # Omega_hat = np.linalg.solve(Sigma_hat, np.eye(len(Sigma_hat)))
        
        ##### Try 1 - Omega #####
        # # Make X_hat conservative thorugh Weighted Factors (III.C of Conservative Edge Sparsification for Graph SLAM Node Removal by Nicholas Carlevaris-Bianco and Ryan M. Eustice)
        # def ApproxWF(w: np.ndarray, LambdaTilde, d):
        #     LambdaWF = np.zeros_like(LambdaTilde)
        #     for i in range(w.size):
        #         LambdaWF[i*d:i*d+d, i*d:i*d+d] = w.item(i) * LambdaTilde[i*d:i*d+d, i*d:i*d+d]
        #         # LambdaWF[i*d:i*d+d, :] = w.item(i) * LambdaTilde[i*d:i*d+d, :]
        #     return LambdaWF

        # def fKL(w, LambdaTilde, LambdaInv: np.ndarray, d):
        #     LambdaWF = ApproxWF(w, LambdaTilde, d)
        #     sign, logdet = np.linalg.slogdet(LambdaWF)
        #     if sign < 0:
        #         print("negative determinant!!!")
        #     ret = np.linalg.trace(LambdaWF @ LambdaInv) - sign * logdet
        #     # print(w, ret)
        #     return ret

        # def MatrixAgtB(A, B):
        #     return -np.sum(A - B <= np.zeros_like(A))

        #     if np.all(A - B >= np.zeros_like(A)):
        #         return 1,
        #     else:
        #         return 0
        
        # def ConstraintPSD(w, LambdaTilde, Lambda, d):
        #     LambdaWF = ApproxWF(w, LambdaTilde, d)
        #     diff = Lambda - LambdaWF
        #     return np.linalg.eigvalsh(diff).min()

        # n = int(len(Omega) / d)
        # w0 = 0.99 * np.ones(n)
        # bounds = [(1e-6,1)] * n
        # # OmegaDiag = ApproxWF(w0, Omega, d)

        # OmegaInv = np.linalg.solve(Omega, np.eye(len(Omega)))
        # # OmegaInvDiag = ApproxWF(w0, np.linalg.solve(Omega, np.eye(len(Omega))), d)
        # # OmegaInvDiag = ApproxWF(w0, np.linalg.inv(Omega), d)
        # # OmegaHatDiag = ApproxWF(w0, Omega_hat, d)
        # constraints = [{
        #     'type': 'ineq',
        #     # 'fun': lambda w: MatrixAgtB(Omega, ApproxWF(w, Omega_hat, d))
        #     'fun': ConstraintPSD, 'args': (Omega_hat, Omega, d)
        # }]

        # result = sp.optimize.minimize(fKL, w0, args=(Omega_hat, OmegaInv, d), bounds=bounds, constraints=constraints, method="SLSQP")
        # wStar = result.x
        # print("w0: ", w0)
        # print("w* = ", wStar)
        # # wStar = 0.5 * np.ones_like(wStar)
        # Omega_hat = ApproxWF(wStar, Omega_hat, d)
        # # print("Omega^-1 == OmegaDiag^-1:\n", np.linalg.inv(Omega) == OmegaInvDiag)
        # # print("trace(Omega_hat @ Omega^-1): ", np.linalg.trace(Omega_hat @ np.linalg.inv(Omega)))
        # # print("trace(Omega_hat @ OmegaDiag^-1): ", np.linalg.trace(Omega_hat @ OmegaInvDiag))
        # # print("Omega:\n", OmegaDiag)
        # # print("Omega_hat:\n", OmegaHatDiag)
        # # print("Omega - OmegaHat:\n", OmegaDiag - OmegaHatDiag)
        # # print("Omega - OmegaHat >= 0:\n", (OmegaDiag - OmegaHatDiag) >= 0)

        # ##### Try 2 - Sigma #####
        # # Make X_hat conservative thorugh Weighted Factors (III.C of Conservative Edge Sparsification for Graph SLAM Node Removal by Nicholas Carlevaris-Bianco and Ryan M. Eustice)
        # def ApproxWF(w: np.ndarray, M_Tilde, d):
        #     M_WF = np.zeros_like(M_Tilde)
        #     for i in range(w.size):
        #         M_WF[i*d:i*d+d, i*d:i*d+d] = w.item(i) * M_Tilde[i*d:i*d+d, i*d:i*d+d]
        #     return M_WF

        # def fKL(w, SigmaTilde, Sigma: np.ndarray, d):
        #     SigmaWF = ApproxWF(w, SigmaTilde, d)
        #     SigmaWF_Inv = np.linalg.solve(SigmaWF, np.eye(SigmaWF.shape[0]))
        #     sign, logdet = np.linalg.slogdet(SigmaWF)
        #     if sign < 0:
        #         print("negative determinant!!!")
        #     ret = np.linalg.trace(SigmaWF_Inv @ Sigma) - sign * logdet
        #     # print(w, ret)
        #     return ret

        # def ConstraintPSD(w, SigmaTilde, Sigma, d):
        #     SigmaWF = ApproxWF(w, SigmaTilde, d)
        #     diff = SigmaWF - Sigma
        #     return np.linalg.eigvalsh(diff).min()
        
        # n = int(len(Sigma) / d)
        # w0 = 0.99 * np.ones(n)
        # bounds = [(1e-6,1)] * n

        # constraints = [{
        #     'type': 'ineq',
        #     'fun': ConstraintPSD, 'args': (Sigma_hat, Sigma, d)
        # }]

        # result = sp.optimize.minimize(fKL, w0, args=(Sigma_hat, Sigma, d), bounds=bounds, constraints=constraints, method="SLSQP")
        # wStar = result.x
        # print("w0: ", w0)
        # print("w* = ", wStar)
        # # wStar = 0.5 * np.ones_like(wStar)
        # Sigma_hat = ApproxWF(wStar, Sigma_hat, d)

        # Omega_hat = np.zeros_like(Sigma_hat)
        # I_d = np.eye(d)
        # for i in range(int(len(Omega_hat) / d)):
        #     Omega_hat[i*d:i*d+d, i*d:i*d+d] = np.linalg.solve(Sigma_hat[i*d:i*d+d, i*d:i*d+d], I_d)


        factorsToAdd = []
        for k, (key_i, key_j, _) in enumerate(chowLiuTree):
            if np.any(np.isnan(block_infos[k])):
                print(f"NaN detected in Omega_hat block for keys {key_i} and {key_j}, falling back to dense factor.")
                return [gtsam.LinearContainerFactor(hessianFactor)]
            noiseModel = gtsam.noiseModel.Gaussian.Information(block_infos[k])
            p_i = values.atPose3(key_i)

            if key_j is None: # Unary factor
                factorsToAdd.append(gtsam.PriorFactorPose3(key_i, p_i, noiseModel))
            else: # Between factor
                p_j = values.atPose3(key_j)
                factorsToAdd.append(gtsam.BetweenFactorPose3(key_i, key_j, p_i.between(p_j), noiseModel))

        if addGpsUnary:
            if rootKey is None:
                rootKey = list(hessianFactor.keys())[0]
            if gps_info is None or np.any(np.isnan(gps_info)):
                print("NaN detected in GPS info block, falling back to dense factor.")
                return [gtsam.LinearContainerFactor(hessianFactor)]
            p_root = values.atPose3(rootKey).translation()
            gpsNoise = gtsam.noiseModel.Gaussian.Information(gps_info)
            factorsToAdd.append(gtsam.GPSFactor(rootKey, p_root, gpsNoise))

        return factorsToAdd




















def main(graphPath:str | None = None, marginalizeType:str="whole", vizualize:bool=True, block_width:float=50.0):
    import os

    fgtools.setBlockWidth(block_width)

    # Load graph
    fileName = "fullSavedGraph.g2o"
    if graphPath is None:
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        graphPath = os.path.join(scriptDir, "../results/second_session/")
        print(f"graphPath: {graphPath}")
        graph, values = fgtools.readG2oWithPrior(graphPath + fileName, loadGPS=False)
    else:
        print(f"graphPath: {graphPath}")
        graph, values = fgtools.readG2oWithPrior(graphPath + fileName)

    


    hasPrior = False
    for i in range(graph.size()):
        f = graph.at(i)
        if isinstance(f, gtsam.PriorFactorPose3):
            hasPrior = True
            break
    if not hasPrior:
        print("No prior factor detected in graph. Adding prior to first node.")
        firstKey = list(values.keys())[0]
        graph.add(gtsam.PriorFactorPose3(firstKey, values.atPose3(firstKey), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]))))

    orig_optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    values = orig_optimizer.optimize()

    originalMarginals = gtsam.Marginals(graph, values)

    # Defines the type of the dictionary instead of [any, any] so that Pylance can autofill and such
    blocks: dict[str, fgtools.BlockStruct] = {}
    blockKeyOrder = []

    pose = {}
    for i, key in enumerate(values.keys()):
        p = values.atPose3(key)

        # Calculate which cell the pose is in
        cell, dist_to_center = fgtools.getBlockCell(p.x(), p.y())
        if i == 0:
            secondPose = values.atPose3(list(values.keys())[1]) # Put the first pose in the same cell as the second pose.
            if np.linalg.norm(np.array([secondPose.x() - p.x(), secondPose.y() - p.y()])) < 5.:
                cell, _ = fgtools.getBlockCell(secondPose.x(), secondPose.y())
                # print(f"Warning: The first two poses are not in the same cell. This may cause issues with the block structure. First pose: ({p.x()}, {p.y()}), Second pose: ({secondPose.x()}, {secondPose.y()})")

        # Create the block 'key' based on grid location
        block_key = fgtools.getBlockKey([cell.x, cell.y])
        if not blockKeyOrder or blockKeyOrder[-1] != block_key:
            blockKeyOrder.append(block_key)
            
        # Update poses and closest to center pose
        if block_key in blocks:
            poses, closestDist, poseKey = blocks[block_key].poses, blocks[block_key].closestDist, blocks[block_key].poseKey
            if (dist_to_center < closestDist):
                closestDist = dist_to_center
                poseKey = key
        else:
            poses = {}
            closestDist = dist_to_center
            poseKey = key
        poses[key] = p
        
        blocks[block_key] = fgtools.BlockStruct(poses, closestDist, poseKey, block_key, cell)

        # break
    internalThreshold = fgtools.ThresholdStruct(0.05, 0.05, 0.05)
    globalThreshold = fgtools.ThresholdStruct(0.1, 0.1, 0.50)

    for block_key in blocks.keys():
        # smallCov = fgtools.isBlockCovSmall(originalMarginals, blocks[block_key].poses, internalThreshold, globalThreshold)
        smallCov = True # FIXME: For now, just assume all blocks have small covariance to test marginalization. We can add the covariance check back in later.
        blocks[block_key].smallCov = smallCov
        print(f"{block_key}, smallCov: {smallCov}")

    # Types: "sequential", "block", "whole"
    # marginalizeType = "block"
    # Marginalize out nodes
    startTime = time.time()
    nfr = GraphSparsification(graph, values, showPlots=False)
    cnt = 0

    print(f"Starting marginalization with type \"{marginalizeType}\"...")
    if marginalizeType == "whole":
        n = len(blocks.keys())
        nfr.showPlot = vizualize

        posesToMarginalize = set(values.keys())
        posesToKeep = []
        for i, blockKey in enumerate(blocks.keys()):
            block = blocks[blockKey]
            if block.smallCov:
                posesToKeep.append(block.poseKey)
            else:
                posesToKeep.extend(block.poses.keys())

        posesToMarginalize -= set(posesToKeep)
        sparse_graph, sparse_values = nfr.MarginalizeNodes(list(posesToMarginalize), marginalizationType="remove", method="nfr_tree", printTime=True)
        
        if vizualize:
            plt.suptitle("Marginalize entire graph at once")
            plt.tight_layout()
            plt.show()

    elif marginalizeType == "block":
        n = len(blocks.keys())
        for i, blockKey in enumerate(blocks.keys()):
            block = blocks[blockKey]
            print(blockKey)

            if block.smallCov:
                posesToMarginalize = set(block.poses.keys())
                posesToMarginalize -= set([block.poseKey])
                if i == (n-1):
                    print("\nFinal Block:")
                    nfr.showPlot = vizualize
                nfr.MarginalizeNodes(list(posesToMarginalize), method="nfr_tree", printTime=False)
            else:
                print(f"Skipping block {blockKey} because it does not have small covariance.")
        sparse_graph, sparse_values = nfr.graph, nfr.values
        if vizualize:
            nfr.PlotGraphDiff()
            plt.suptitle("Optimization after every block marginalization")
            plt.tight_layout()
            plt.show()
                

    elif marginalizeType == "sequential":
        for i, k in enumerate(values.keys()):
            blockKey = fgtools.getBlockKey(values.atPose3(k))

            if (blocks[blockKey].smallCov and k != blocks[blockKey].poseKey) or (i == (len(values.keys()) - 1)):
                print(i)

                if i == (len(values.keys()) - 1):
                # if cnt >= 100 or i == (len(values.keys()) - 1):
                    nfr.showPlot = vizualize
                    nfr.MarginalizeNodes([k], printTime=False)
                    nfr.showPlot = False
                    cnt = 0
                    plt.show()
                else:
                    nfr.MarginalizeNodes([k], printTime=False)
                
                cnt +=1
        sparse_graph, sparse_values = nfr.graph, nfr.values
        if vizualize:
            plt.suptitle("Optimization after every pose marginalization")
            plt.tight_layout()
            plt.show()
    
    else:
        # raise ValueError(f"marginalizeType must be whole, block, or sequential, not \"{marginalizeType}\"")
        print((f"marginalizeType must be whole, block, or sequential, not \"{marginalizeType}\""))
    
    print(f"Total time for marginalization: {time.time() - startTime}")

    # Ask user if they want to save the graph
    save = input("Do you want to save the marginalized graph? (y/[n]): ")
    if save.lower() == "y":
        savePath = input("Enter the path to save the graph (just the directory, e.g. \"../TestDir/\"): ")
        if not savePath.endswith("/"):
            savePath += "/"

        pcds = fgtools.loadPosePCDs(graphPath + "Scans/")
        block_pcds, non_block_pcds = fgtools.combinePCDsInBlocks(blocks, pcds, values)

        new_graph, new_values, pose2block_mapping, key_mapping = fgtools.rekeyGraphAndBlocks(sparse_graph, sparse_values, blocks)

        
        fgtools.savePointClouds(block_pcds, non_block_pcds, savePath, pose2block_mapping, key_mapping, removeExisting=True)

        fgtools.writeG2oWithPrior(new_graph, new_values, savePath + "sparseGraph.g2o")
        print(f"Graph saved to {savePath}")
    return graph, values, blocks, nfr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GraphSparsification marginalization on a g2o graph.")
    parser.add_argument("--graphPath", type=str, default=None, help="Path to the directory containing the g2o graph and scans. If not provided, defaults to '../results/largeFG/'.")
    parser.add_argument("--marginalizeType", type=str, default="whole", help="Type of marginalization to perform: 'whole', 'block', or 'sequential'. Defaults to 'whole'.")
    parser.add_argument("--viz", "-v", action="store_true", help="Display visualization of the marginalization process.")
    parser.add_argument("--blockWidth", "-bw", type=float, default=50.0, help="Width of blocks for block marginalization. Only used if marginalizeType is 'block'. Defaults to 50.0.")

    args = parser.parse_args()
    main(graphPath=args.graphPath, marginalizeType=args.marginalizeType, vizualize=args.viz, block_width=args.blockWidth)