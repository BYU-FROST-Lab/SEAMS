import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=False)
import os
import open3d as o3d
import copy
from typing import Any
from pathlib import Path

# Local imports
import gtsam

BLOCK_WIDTH:float = 50.0

def setBlockWidth(width:float):
    global BLOCK_WIDTH
    BLOCK_WIDTH = width

def getBlockWidth():
    return BLOCK_WIDTH

class GridIndex:
    def __init__(self, x, y):
        self.x = int(x // BLOCK_WIDTH) + 1 if x >= 0 else int(x / BLOCK_WIDTH) - 1
        self.y = int(y // BLOCK_WIDTH) + 1 if y >= 0 else int(y / BLOCK_WIDTH) - 1
    
    def print(self):
        print(f"GridIndex: x={self.x}, y={self.y}")

# class Space3D:
#     def __init__(self, x, y, z=None):
#         self.x = x
#         self.y = y
#         self.z = z

# class GridStruct:
#     def __init__(self, x, y):
#         gridIndex = GridIndex(x, y)
#         center = 

class BlockStruct:
    def __init__(self, poses: dict, closestDist: float, poseKey: int, blockKey: str, cell, smallCov: bool=False):
        self.poses = poses
        self.closestDist = closestDist
        self.poseKey = poseKey
        self.blockKey = blockKey
        if type(cell) == GridIndex:
            self.grid_idx = cell
        else:
            self.grid_idx = (GridIndex(cell[0], cell[1]))
        self.hasFactorWithOtherBlocks = []
        self.smallCov = smallCov

class ThresholdStruct:
    def __init__(self, x: float, y: float, z: float):
        '''
        Inputs:
        x, y, and z inputs are given in 99.5% confidence notation. 
        i.e. 3 * sigma
        Returns:
        x, y, and z thresholds are the variances relating to the 3 * sigma inputs
        i.e. (input / 3)**2
        '''
        self.x = self.toVarFrom3Sigma(x)
        self.y = self.toVarFrom3Sigma(y)
        self.z = self.toVarFrom3Sigma(z)

    def toVarFrom3Sigma(self, input):
        return (input / 3.0) * (input / 3.0)
    
    def det(self):
        return np.linalg.det(np.diag([self.x, self.y, self.z]))

def add_generic_prior(key , graph: gtsam.gtsam.NonlinearFactorGraph, values: gtsam.gtsam.Values):
    """
    :param key: Description
    :param graph: Description
    :type graph: gtsam.gtsam.NonlinearFactorGraph
    :param values: Description
    :type values: gtsam.gtsam.Values

    :return graph: New graph with added prior on key
    """
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    graph.addPriorPose3(key, values.atPose3(key), prior_noise)
    print("Added generic prior to key {}".format(key))
    return graph


def load_graph(directory: str, fileName: str, addGenericPrior: bool = True):
    fullPath = os.path.join(directory, fileName+".g2o")
    graph, values = gtsam.readG2o(fullPath, True)
    if addGenericPrior:
        graph = add_generic_prior(list(values.keys())[0], graph, values)
    return graph, values


def plot_graph(graph: gtsam.gtsam.NonlinearFactorGraph, 
               values: gtsam.gtsam.Values, 
               ax:Axes|None=None, 
               color_override:str|None=None, 
               nodeColor:str="tab:blue", 
               edgeColor:str="default", 
               alpha:float|None=None, 
               title:str|None=None,
               addKeyLabels:bool=False,
               changeColorBySymbols:bool=False,
               symbolColors: dict[int, str] = {},
               plotValuePoints:bool=True,
               ):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if isinstance(color_override, str):
        unary_factor_color = color_override
        linearContainer_factor_color = color_override
        default_factor_color = color_override
        default_point_color = color_override
        if alpha is None:
            alpha = 0.21
    else:
        if alpha is None:
            alpha = 1.
        if edgeColor == "default":
            unary_factor_color = "tab:green"
            linearContainer_factor_color = "tab:purple"
            default_factor_color = "tab:orange"
        else:
            unary_factor_color = edgeColor
            linearContainer_factor_color = edgeColor
            default_factor_color = edgeColor
        default_point_color = nodeColor
    
    if not isinstance(title, str):
        title = ""

    # ------------------------------------------------------------
    # Draw edges
    # ------------------------------------------------------------
    N = graph.size()
    global_cx = []
    global_cy = []
    global_c_color = []
    global_segments = []
    global_seg_color = []
    for i in range(N):
        factor = graph.at(i)

        # Only plot BetweenFactorPose3
        # if isinstance(factor, gtsam.BetweenFactorPose3):
        if factor is None:
            continue
        keys = factor.keys()
        # if len(keys) == 1: # Only one key means this is a prior, skip for now # FIXME
        #     continue
        x = []
        y = []
        for k in keys:
            if k not in values.keys():
                raise Exception(f"plot_graph: key {k} is not found in the provided poses")
            p = values.atPose3(k)
            x.append(p.x())
            y.append(p.y())
        
        if len(keys) == 1: # Unary factor (prior or gps or something)
            cx = x[0]
            cy = y[0] + 5
            factor_color = unary_factor_color
        elif isinstance(factor, gtsam.LinearContainerFactor):
            cx = np.mean(x)
            cy = np.mean(y)
            factor_color = linearContainer_factor_color
        else:
            cx = np.mean(x)
            cy = np.mean(y)
            factor_color = default_factor_color
        
        global_cx.append(cx)
        global_cy.append(cy)
        global_c_color.append(factor_color)

        for j in range(len(x)):
            global_segments.append([(x[j], y[j]),(cx, cy)])
            global_seg_color.append(factor_color)
            
            
    ax.scatter(global_cx, global_cy, marker='s', s=1, color=global_c_color, alpha=alpha)
    lc = LineCollection(global_segments, colors=global_seg_color, linewidths=0.5, alpha=alpha)
    ax.add_collection(lc)
        
        
        
    # ------------------------------------------------------------
    # Plot poses
    # ------------------------------------------------------------
    if plotValuePoints:
        plot_points(values.keys(), values, ax=ax, color=default_point_color, alpha=alpha, addKeyLabels=addKeyLabels, 
                    changeColorBySymbols=changeColorBySymbols, symbolColors=symbolColors)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')
    ax.xaxis.set_major_locator(MultipleLocator(BLOCK_WIDTH))
    ax.yaxis.set_major_locator(MultipleLocator(BLOCK_WIDTH))
    ax.grid(color='lightgray',linestyle='--')
    # plt.show()
    return ax

def plot_points(
        keys, 
        values: gtsam.gtsam.Values, 
        color="tab:blue", 
        alpha=1., 
        ax=None, 
        addKeyLabels=False, 
        changeColorBySymbols:bool=False,
        symbolColors: dict[int, str] = {},
        ):
    px = []
    py = []
    keyLabels = []
    colors = []


    if ax is None:
        ax = plt.gca()

    nextColorInt = 0
    for k in keys:
        p = values.atPose3(k)
        px.append(p.x())
        py.append(p.y())
        s = gtsam.Symbol(k)
        if s.chr():
            keyLabels.append(s.string())
        else:
            keyLabels.append(str(k))
        
        if s.chr() not in symbolColors:
            while True:
                newColor = f'C{nextColorInt}'
                if newColor in symbolColors.values():
                    nextColorInt+=1
                else:
                    symbolColors[s.chr()] = newColor
                    nextColorInt+=1
                    break

        if changeColorBySymbols:
            colors.append(symbolColors[s.chr()])
        else:
            colors.append(color)
            
    ax.scatter(px, py, s=10, color=colors, alpha=alpha)

    if addKeyLabels:
        for x, y, k in zip(px, py, keyLabels):
            ax.annotate(k, (x, y), xytext=(5,5), textcoords="offset points")

    # Add Legend
    if changeColorBySymbols:
        legendElements = []
        for ascii_chr, color in symbolColors.items():
            legendElements.append(
                    Line2D([0],[0], marker='o', color='w', label=chr(ascii_chr), markerfacecolor=color),
            )
        ax.legend(handles=legendElements, loc='lower right')

def plot_isam_graph(isam: gtsam.ISAM2):
    curr_est = isam.calculateEstimate()
    curr_graph = isam.getFactorsUnsafe()
    plot_graph(curr_graph, curr_est)

def plot_graph_covariances(keys: list[int], 
                           graph: gtsam.NonlinearFactorGraph | gtsam.IncrementalFixedLagSmoother, 
                           values: gtsam.Values, 
                           ax, 
                           subset='xy', 
                           n_std=3.0, 
                           changeColorBySymbols:bool=False,
                           symbolColors: dict[int, str] = {}, 
                           **kwargs):
    '''
    Optional keyword arguments are: [kw: default_arg]
    edgecolor: 'tab:blue'
    linestyle: '-'
    '''
    if isinstance(graph, gtsam.NonlinearFactorGraph):
        try:
            marginals = gtsam.Marginals(graph, values)
        except Exception as e:
            print("Error creating marginals: ", e)
            return
    elif isinstance(graph, gtsam.IncrementalFixedLagSmoother):
        marginals = graph
    else:
        raise TypeError("graph must be a gtsam.NonlinearFactorGraph or gtsam.IncrementalFixedLagSmoother")

    patches = []
    
    # Pre-define styles to avoid dict merging in the loop
    edgecolor = kwargs.get('edgecolor', 'tab:blue')
    linestyle = kwargs.get('linestyle', '-')
    alpha = kwargs.get('alpha', 0.7)
    facecolor = kwargs.get('facecolor', 'none')

    edgeColors = []

    # Slice indices mapping to avoid if/else overhead inside the loop
    slices = {
        'rp': (0, 2, 1), 'py': (1, 3, 1), 'ry': (0, 3, 2),
        'xy': (3, 5, 1), 'yz': (4, 6, 1)
    }
    
    nextColorInt = 0
    for k in keys:
        # 1. Get Covariance
        full_cov = marginals.marginalCovariance(k)
        
        # 2. Extract subset (Manual slicing is faster than strings)
        if subset == 'xy':
            cov = full_cov[3:5, 3:5]
        elif subset == 'xz':
            cov = full_cov[3:6:2, 3:6:2]
        else:
            # Add other slices here...
            cov = full_cov[3:5, 3:5] 

        # 3. Calculate Ellipse Params via Eigenvalues (Faster than Affine Transforms)
        # This replaces the entire covariance_ellipse function logic
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are 2 * n_std * sqrt(eigenvalues)
        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))

        # 4. Get Pose
        pose = values.atPose3(k)
        
        # Create the patch (don't add to axes yet)
        ell = Ellipse(xy=(pose.x(), pose.y()), width=width, height=height, angle=theta)
        patches.append(ell)
        
        s = gtsam.Symbol(k)
        if s.chr() not in symbolColors:
            while True:
                newColor = f'C{nextColorInt}'
                if newColor in symbolColors.values():
                    nextColorInt+=1
                else:
                    symbolColors[s.chr()] = newColor
                    break
        
        if changeColorBySymbols:
            edgeColors.append(symbolColors[s.chr()])
        else:
            edgeColors.append(edgecolor)


    # 5. Add everything in one batch
    collection = PatchCollection(patches, 
                                 edgecolor=edgeColors, 
                                 facecolor=facecolor, 
                                 linestyle=linestyle, 
                                 alpha=alpha,
                                 match_original=False) # Match original ignores individual patch styles for speed
    ax.add_collection(collection)

def plot_graph_covariances_original(keys, graph, values, ax, subset='xy', **kwargs):
    """
    Plot the covariances for certain keys associated with a graph.

    :param subset: Slice of ellipse that you want to see. Options are 'xy', 'yz', 'xz'
    :type subset: str
    :param kwargs: Some of the possible kwargs: edgecolor='tab:blue', linestyle="--", alpha=0.7
    """
    try:
        marginals = gtsam.Marginals(graph, values)
    except Exception as e:
        print("Error creating marginals: ", e)
        return
    for k in keys: 
        cov = marginals.marginalCovariance(k)
        if subset == 'rp': # roll and pitch
            cov = cov[:2,:2]
        elif subset == 'py': # pitch and yaw
            cov = cov[1:3,1:3]
        elif subset == 'ry': # roll yaw
            cov = cov[:3:2,:3:2]
        elif subset == 'xy':  # x and y
            cov = cov[3:5,3:5]
        elif subset == 'yz':  # y and z
            cov = cov[4:6,4:6]
        elif subset == 'xz': # x and z
            cov = cov[3:6:2,3:6:2] 
        else:
            raise ValueError(f"subset must be 'rp', 'py', 'ry', 'xy', 'yz', or 'xz' but current value is {subset}")

        pose = values.atPose3(k)
        covariance_ellipse(pose.x(), pose.y(), cov, ax, **kwargs)


# The following function is copied from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
# and then modified for my use.
def covariance_ellipse(x, y, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse.

    Parameters
    ----------
    x, y : float
        Center of ellipse

    cov : array-like, shape (2, 2)
        Input 2D covariance.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if np.shape(cov) != (2, 2):
        raise Exception(f"covariance_ellipse: cov must have shape (2, 2). Current shape {np.shape(cov)}")
    default_kwargs = {
        "edgecolor": 'tab:blue', 
        "linestyle": "-", 
        "alpha": 0.7
    }
    merged_config = {**default_kwargs, **kwargs}

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **merged_config)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def getBlockKey(cell):
    if type(cell) == GridIndex:
        cx = cell.x
        cy = cell.y
    elif type(cell) == gtsam.Pose3:
        cell, _ = getBlockCell(cell.x(), cell.y())
        cx = cell.x
        cy = cell.y
    else:
        cx = int(cell[0])
        cy = int(cell[1])
    if cx == 0:
        cx +=1
    if cy == 0:
        cy +=1
    return f"block_{cx}_{cy}"

def getBlockCell(x, y):

    cell = GridIndex(x, y)
    signx = 1 if cell.x >= 0 else -1
    signy = 1 if cell.y >= 0 else -1

    cx = (cell.x - signx * 0.5) * BLOCK_WIDTH
    cy = (cell.y - signy * 0.5) * BLOCK_WIDTH

    dx = cx - x
    dy = cy - y
    dist = dx * dx + dy * dy
    return cell, dist

def getRelativeCov(pose1: gtsam.gtsam.Pose3, pose2: gtsam.gtsam.Pose3, jointCov):
    '''
    Calculates the relative covariance between two poses as described in equation (24)
        in this gtsam post https://gtsam.org/2021/02/23/uncertainties-part3.html
    '''
    if np.shape(jointCov) != (12, 12):
        raise Exception(f"Joint covariance has wrong shape: {np.shape(jointCov)}")
    
    Ad_p1 = pose1.AdjointMap()
    Ad_p2_inv = pose2.inverse().AdjointMap()
    Ad_comb = Ad_p2_inv @ Ad_p1

    cov11 = jointCov[:6,:6] 
    cov22 = jointCov[6:,6:]
    cov12 = jointCov[:6,6:]

    relCov = Ad_comb @ cov11 @ Ad_comb.T + cov22 - Ad_comb @ cov12 - cov12.T @ Ad_comb.T
    # print("World Reference:")
    # print(cov22)
    # print("Pose1 Reference:")
    # print(relCov)
    return relCov


def isBlockCovSmall(marginals: gtsam.Marginals, poses, internalThreshold: ThresholdStruct, globalThreshold: ThresholdStruct):
    isInternalSmall = True
    isGlobalSmall = True
    poseKeys = list(poses.keys())
    keys_to_check = [poseKeys[0], poseKeys[-1]]
    for i in range(1, len(poseKeys)):
        if (poseKeys[i] - poseKeys[i - 1]) != 1:
            keys_to_check.append(poseKeys[i-1])
            keys_to_check.append(poseKeys[i])
    # print(keys_to_check)

    for i in range(len(keys_to_check) - 1):
        # Global Covariance
        iCov = marginals.marginalCovariance(keys_to_check[i])
        det_i = np.linalg.det(iCov[:3,:3])
        iVar = np.diag(iCov)
        if iVar[0] > globalThreshold.x or iVar[1] > globalThreshold.y or iVar[2] > globalThreshold.z or det_i > globalThreshold.det():
            print(f"Global covariance too big: key \"{keys_to_check[i]}\"")
            print(f"  Global threshold: {globalThreshold.x}, {globalThreshold.y}, {globalThreshold.z}, det: {globalThreshold.det()}")
            print(f"  Covariance: {iVar}, det: {det_i}")
            return False

        # Internal Covariance
        for j in range(i+1, len(keys_to_check)):
            if keys_to_check[i] == keys_to_check[j]:
                continue
            # print(f"Checking relative covariance between keys \"{keys_to_check[i]}\" and \"{keys_to_check[j]}\"")
            jointCov = marginals.jointMarginalCovariance([keys_to_check[i], keys_to_check[j]]).fullMatrix()
            # print("Joint covariance:\n", jointCov)
            relCov = getRelativeCov(poses[keys_to_check[i]], poses[keys_to_check[j]], jointCov)
            det_rel = np.linalg.det(relCov[:3,:3])
            relVar = np.diag(relCov)
            if relVar[0] > internalThreshold.x or relVar[1] > internalThreshold.y or relVar[2] > internalThreshold.z or det_rel > internalThreshold.det():
                print(f"Relative covariance too big between keys \"{keys_to_check[i]}\" and \"{keys_to_check[j]}\"")
                print(f"relVal > Threshold")
                print(f"x: {relVar[0]} > {internalThreshold.x} -> {relVar[0] > internalThreshold.x}")
                print(f"y: {relVar[1]} > {internalThreshold.y} -> {relVar[1] > internalThreshold.y}")
                print(f"z: {relVar[2]} > {internalThreshold.z} -> {relVar[2] > internalThreshold.z}")
                print(f"det: {det_rel} > {internalThreshold.det()} -> {det_rel > internalThreshold.det()}")
                return False    
    
    return True

def factor_exists_between(graph: gtsam.gtsam.NonlinearFactorGraph, key1, key2):
    for i in range(graph.size()):
        factor = graph.at(i)
        f_keys = factor.keys()
        if key1 in f_keys and key2 in f_keys:
            return True
    return False

def remove_priors(graph: gtsam.NonlinearFactorGraph):
    new_graph = gtsam.NonlinearFactorGraph()

    for i in range(graph.size()):
        f = graph.at(i)
        # if i == 0:
        #     print(f)
        
        # Check if the factor is a prior
        if isinstance(f, gtsam.PriorFactorPose3):
            # print(f"Removing prior at for factor {f}")
            # print(f"Factor noise is {f.noiseModel()}")
            continue

        new_graph.push_back(f)
    
    return new_graph

def createSubGraph(graph: gtsam.NonlinearFactorGraph, values: gtsam.Values, keys_to_keep: list|set, 
                originPoseKey=None, sub_graph: gtsam.NonlinearFactorGraph = None, sub_values: gtsam.Values = None):
    """
    Create sub graph given desired keys to keep

    :param graph: Original factor graph to pull sub graph from
    :type graph: gtsam.NonlinearFactorGraph
    :param values: Values corresponding to original factor graph
    :type values: gtsam.Values
    :param keys_to_keep: list of keys to the desired nodes to keep
    :param originPoseKey: Optional. Key of the pose to set to the origin
    :param sub_graph: Optional. Can pass in an existing graph and this will add the subgraph to the existing one.
    :type sub_graph: gtsam.NonlinearFactorGraph
    :param sub_values: Optional. Same idea as sub_graph
    :type sub_values: gtsam.Values
        
    Return:
        New sub graph and values (return sub_graph, sub_values)

    """
    if sub_graph is None:
        sub_graph = gtsam.NonlinearFactorGraph()
    if sub_values is None:
        sub_values = gtsam.Values()

    if originPoseKey is not None:
        originPose = values.atPose3(originPoseKey)
    else:
        originPose = gtsam.Pose3.Identity()

    keys_to_keep_set = set(keys_to_keep)
    for i in range(graph.size()):
        f = graph.at(i)

        f_keys_set = set(f.keys())
        # if len(f_keys_set) > 2:
        #     print(f_keys_set)
        if not f_keys_set - keys_to_keep_set: # If all of the factor keys are in keys_to_keep, then the set will be empty and this will be true
            sub_graph.push_back(f)
    
    for key in keys_to_keep:
        if values.exists(key):
            tmpValue = values.atPose3(key)
            sub_values.insert(key, gtsam.Pose3(tmpValue.rotation(), tmpValue.translation()-originPose.translation()))
    
    return sub_graph, sub_values


def getRepresentativeFactor(key1, key2, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values, plot=False) -> gtsam.BetweenFactorPose3:
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-9]*6)
    shifted = gtsam.Values()

    pose1 = values.atPose3(key1)
    pose2 = values.atPose3(key2)

    for key in values.keys():
        tmpPose = values.atPose3(key)
        shifted.insert(key, gtsam.Pose3(tmpPose.rotation(), tmpPose.translation()-pose1.translation()))

    # graph_shifted, shifted = createSubGraph(graph, values, keys, key1)


    graph_shifted = remove_priors(graph)
    graph_shifted.addPriorPose3(key1, shifted.atPose3(key1), prior_noise)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph_shifted, shifted)
    shifted = optimizer.optimize()

    marginals_shifted = gtsam.Marginals(graph_shifted, shifted)

    # print("Pose3:")
    # print(shifted.atPose3(key2))
    # print("Covariance:")
    # print(marginals_shifted.marginalCovariance(key2))

    pose1_shifted = shifted.atPose3(key1)
    pose2_shifted = shifted.atPose3(key2)

    factor_noise = gtsam.noiseModel.Gaussian.Covariance(marginals_shifted.marginalCovariance(key2))
    between_factor = gtsam.BetweenFactorPose3(key1, key2, pose1_shifted.between(pose2_shifted), factor_noise)

    if plot:
        graph_shifted.add(between_factor)
        plot_graph(graph_shifted, shifted)
        plt.scatter(pose1.x(), pose1.y(), color='red')
        plt.scatter(pose2.x(), pose2.y(), color='green')

        plt.scatter(pose1_shifted.x(), pose1_shifted.y(), color='red')
        plt.scatter(pose2_shifted.x(), pose2_shifted.y(), color='green')

    return between_factor

def getRepresentativeFactorNoise(key1, key2, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values, plot=False) -> gtsam.gtsam.noiseModel.Gaussian:
    between_factor = getRepresentativeFactor(key1, key2, graph, values, plot)
    return between_factor.noiseModel()

def choleskyInverse(A: np.ndarray):
    # Python implementation of Cholesky decomp to get A_inv
    L = np.linalg.cholesky(A)
    A_inv = np.linalg.solve(L.conj().T, np.linalg.solve(L, np.eye(A.shape[0])))
    return A_inv

def isPSD(M, printDebug: bool=True):
    M = np.asarray(M)
    if not np.isclose(M, M.conj()).all():
        if printDebug:
            print("Matrix is not PSD because it is not Hermitian")
        return False

    eigval = np.linalg.eigvalsh(M).min()

    if not (eigval > -1e-5): # Smallest eigenvalue is not positive
        if printDebug:
            print("Smallest eigenvalue is not positive: ", eigval)
        return False

    return True

def checkConservitiveness(graph1: gtsam.gtsam.NonlinearFactorGraph, values1:gtsam.gtsam.Values, graph2: gtsam.gtsam.NonlinearFactorGraph, values2:gtsam.gtsam.Values):
    """
    Check if graph2 is conservative compared against graph1 by comparing
    the diagonal elements of the corresponding covariance matrices.

    :param graph1: Original graph to compare against (true graph)
    :param values1: Values for graph1 (true values)
    :param graph2: New graph
    :param values2: New values

    :return isConservative: True if each element of the diagonal of cov of graph2 is greater than the covariance of the diagonals of graph1.
    """
    isConservative = True

    # sigFigs = lambda x: float(f"{x:.6g}")

    keysToCompare = set(values1.keys()) & set(values2.keys())

    marginals1 = gtsam.Marginals(graph1, values1)
    marginals2 = gtsam.Marginals(graph2, values2)
    for key in keysToCompare:
        cov1 = marginals1.marginalCovariance(key)
        cov2 = marginals2.marginalCovariance(key)

        print(f"\nKey: {key}")
        # print("condition num of cov1: ", np.linalg.cond(cov1))
        # print("condition num of cov2: ", np.linalg.cond(cov2))

        try:
            np.linalg.cholesky(cov2 - cov1)
            print("Is PSD as per numpy cholesky")
        except:
            print("Not PSD as per numpy cholesky")

        if not isPSD(cov2 - cov1):
            psd = isPSD(cov2[:2,:2] - cov1[:2,:2])
            if not psd:
                print("cov1:\n", cov2[:2,:2])
                print("cov2:\n", cov1[:2,:2])
            print("PSD: rp ", psd)
            psd = isPSD(cov2[1:3,1:3] - cov1[1:3,1:3])
            if not psd:
                print("cov1:\n", cov2[1:3,1:3])
                print("cov2:\n", cov1[1:3,1:3])
            print("PSD: py", psd)
            psd = isPSD(cov2[1:3:2,1:3:2] - cov1[1:3:2,1:3:2])
            if not psd:
                print("cov1:\n", cov2[1:3:2,1:3:2])
                print("cov2:\n", cov1[1:3:2,1:3:2])
            print("PSD: ry ", psd)
            psd = isPSD(cov2[3:5,3:5] - cov1[3:5,3:5])
            if not psd:
                print("cov1:\n", cov2[3:5,3:5])
                print("cov2:\n", cov1[3:5,3:5])
            print("PSD: xy ", psd)
            psd = isPSD(cov2[4:6,4:6] - cov1[4:6,4:6])
            if not psd:
                print("cov1:\n", cov2[4:6,4:6])
                print("cov2:\n", cov1[4:6,4:6])
            print("PSD: yz ", psd)
            psd = isPSD(cov2[3:6:2,3:6:2] - cov1[3:6:2,3:6:2])
            if not psd:
                print("cov1:\n", cov2[3:6:2,3:6:2])
                print("cov2:\n", cov1[3:6:2,3:6:2])
            print("PSD: xz ", psd)
        else:
            print("cov2 - cov1 is PSD")
    
    return isConservative

def readG2oWithPrior(filename: str, loadGPS: bool=False):
    """
    Does gtsam.readG2o(filename, is3D=True) followed by reading
    and placing any 3D priors found in file to same graph.
    Only adds priors as type gtsam.PriorFactorPose3

    :param graph: Graph to add prior to
    :param filename: name of the .g2o file "path/example.g2o"
    :param symbolChar: character for key in graph and values

    :return: graph with any priors found and values
    """
    graph, values = gtsam.readG2o(filename, is3D=True)
    with open(filename) as f:
        for line in f:
            words = line.split()
            if words[0] == "EDGE_SE3_PRIOR:QUAT":
                key = int(words[1])
                var = [float(word) for word in words[2:]]
                p = gtsam.Point3(x=var[0], y=var[1], z=var[2])
                r = gtsam.Rot3(w=var[6], x=var[3], y=var[4], z=var[5])
                pose = gtsam.Pose3(r, p)
                infoG2o = np.array([[var[7],  var[8],  var[9],  var[10], var[11], var[12]],
                                [var[8],  var[13], var[14], var[15], var[16], var[17]],
                                [var[9],  var[14], var[18], var[19], var[20], var[21]],
                                [var[10], var[15], var[19], var[22], var[23], var[24]],
                                [var[11], var[16], var[20], var[23], var[25], var[26]],
                                [var[12], var[17], var[21], var[24], var[26], var[27]]])
                
                # Change to gtsam [rotation, translation] instead of the g2o [translation, rotation] 
                infoGtsam = np.zeros_like(infoG2o)
                infoGtsam[:3,:3] = infoG2o[3:,3:]
                infoGtsam[3:,3:] = infoG2o[:3,:3]
                infoGtsam[:3,3:] = infoG2o[3:,:3]
                infoGtsam[3:,:3] = infoG2o[:3,3:]

                noise = gtsam.noiseModel.Gaussian.Information(infoGtsam)
                prior = gtsam.PriorFactorPose3(key, pose, noise)
                # prior.print()
                graph.add(prior)
            elif words[0] == "EDGE_SE3_GPS" and loadGPS:
                key = int(words[1])
                var = [float(word) for word in words[2:]]
                p = gtsam.Point3(x=var[0], y=var[1], z=var[2])
                infoTrans = np.array([[var[7],  var[8],  var[9] ],
                                      [var[8],  var[13], var[14]],
                                      [var[9],  var[14], var[18]]])
                
                noise = gtsam.noiseModel.Gaussian.Information(infoTrans)
                gpsFactor = gtsam.GPSFactor(key, p, noise)
                graph.add(gpsFactor)
    
    return graph, values


def writeG2oWithPrior(graph, values, filename: str, saveGPS:bool=True):
    """
    Do typical gtsam.writeG2o followed by explicitly saving the priors to the same file.
    Save priors from graph to end of .g2o file. 
    Only saves priors of type gtsam.PriorFactorPose3

    :param graph: Factor graph
    :param filename: Name of .g2o file "path/example.g2o"

    :return tuple: (priorCount, gpsCount) number of prior and gps factors found
    """

    gtsam.writeG2o(graph, values, filename)

    priorCnt = 0
    gpsCnt = 0
    with open(filename, "a") as f:
        for i in range(graph.size()):
            factor = graph.at(i)
            if isinstance(factor, gtsam.PriorFactorPose3):
                print("Found Prior")
                sf6 = lambda x: f"{x:.6g}"

                q = factor.prior().rotation().toQuaternion()
                p = factor.prior()
                info = factor.noiseModel().information()
                fk = factor.keys()
                s = f"EDGE_SE3_PRIOR:QUAT {fk[0]} {sf6(p.x())} {sf6(p.y())} {sf6(p.z())} {sf6(q.x())} {sf6(q.y())} {sf6(q.z())} {sf6(q.w())}"

                # Change to [translation, rotation] instead of the gtsam [rotation, translation]
                infoG2o = np.zeros_like(info)
                infoG2o[:3,:3] = info[3:,3:]
                infoG2o[3:,3:] = info[:3,:3]
                infoG2o[:3,3:] = info[3:,:3]
                infoG2o[3:,:3] = info[:3,3:]

                for i in range(6):
                    for j in range(i,6):
                        s += " " + sf6(infoG2o[i,j])
                s += "\n"
                f.write(s)
                priorCnt += 1
            ## There is no noiseModel() binding in python currently and thus cannot save GPS factors for now.
            # elif isinstance(factor, gtsam.GPSFactor) and saveGPS:
            #     print("Found GPS")
            #     sf6 = lambda x: f"{x:.6g}"

            #     p = factor.measurementIn()
            #     info = factor.noiseModel().information()
            #     fk = factor.keys()
            #     s = f"EDGE_SE3_GPS {fk[0]} {sf6(p.x())} {sf6(p.y())} {sf6(p.z())} 0. 0. 0. 1."

            #     # Change to [translation, rotation] instead of the gtsam [rotation, translation]
            #     infoG2o = np.eye(6)
            #     infoG2o[:3,:3] = info

            #     for i in range(6):
            #         for j in range(i,6):
            #             s += " " + sf6(infoG2o[i,j])
            #     s += "\n"
            #     f.write(s)
            #     gpsCnt += 1
    return (priorCnt, gpsCnt)

def rekeyGraphAndBlocks(graph, values, blocks: dict[str, BlockStruct]) \
    -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, dict[int, str], dict[int, int]]:
    '''
    Maps sparse graph and values back to start at index 0 and increase by 1 instead of keeping previous values.
    Modifies blocks in place
    Creates new graph and values
    Returns new_graph, new_values, and poseToBlock dict
    '''
       
    key_mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(values.keys()))}

    pose2block_mapping = {}

    for blockKey in blocks:
        centerKey = blocks[blockKey].poseKey
        blocks[blockKey].poseKey = key_mapping[centerKey]

        if blocks[blockKey].smallCov:
            pose2block_mapping[blocks[blockKey].poseKey] = blockKey

    new_graph = gtsam.NonlinearFactorGraph()
    for i in range(graph.size()):
        f = graph.at(i)
        new_keys = [key_mapping[k] for k in f.keys()]
        new_f = f.rekey(new_keys)
        new_graph.push_back(new_f)

    new_values = gtsam.Values()
    for old_key, new_key in key_mapping.items():
        new_values.insert(new_key, values.atPose3(old_key))

    return new_graph, new_values, pose2block_mapping, key_mapping


def loadPosePCDs(directory: str) -> dict[int, o3d.geometry.PointCloud]:
    pcds = {}

    # Sort files by number and then load
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.pcd')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for file in files:

        # Get rid of leading zeros in the key if they exist
        key_str = ''.join(filter(str.isdigit, file))
        key = int(key_str) if key_str else 0 # This key should match the value keys in the graph
        # print(f"Key for {file} is {key}")
        pcd = o3d.io.read_point_cloud(os.path.join(directory, file))
        pcds[key] = pcd

    return pcds

def loadBlockPCDs(directory: str) -> dict[str, o3d.geometry.PointCloud]:
    block_pcds = {}

    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.pcd')]
    for file in files:
        blockKey = file[:-4] # Remove .pcd extension
        pcd = o3d.io.read_point_cloud(os.path.join(directory, file))
        block_pcds[blockKey] = pcd

    return block_pcds

def combinePCDsInBlocks(blocks: dict[str, BlockStruct], pcds: dict[int, o3d.geometry.PointCloud], values: gtsam.Values) \
                        -> tuple[dict[str, o3d.geometry.PointCloud], dict[int, o3d.geometry.PointCloud]]:
    '''
    Blocks, pcds, and values should all be in the same key space (i.e. if a block has poseKey 5, then there should be a pcd with key 5 and a value with key 5)
    Also, the values needs to be prior to sparsification so that there is a pose for each pcd key.
    '''
    block_pcds = {}
    non_block_pcds = {}
    for blockKey in blocks:
        if blocks[blockKey].smallCov:
            block = blocks[blockKey]
            block_pcd = o3d.geometry.PointCloud()
            for poseKey in block.poses:
                if poseKey in pcds:
                    # Transform the pcd to the coordinate frame of the pose
                    pcd = pcds[poseKey]
                    pose = values.atPose3(poseKey)
                    pcd.transform(pose.matrix())
                    block_pcd += pcd
                else:
                    print(f"Pose key {poseKey} in block {blockKey} not found in pcds")
            # Transform the block pcd to the coordinate frame of the block center
            centerPose = values.atPose3(block.poseKey)
            block_pcd.transform(centerPose.inverse().matrix())
            block_pcds[blockKey] = block_pcd
        else:
            for poseKey in blocks[blockKey].poses:
                if poseKey in pcds:
                    # Transform the pcd to the coordinate frame of the pose
                    pcd = pcds[poseKey]
                    # pose = values.atPose3(poseKey)
                    # pcd.transform(pose.matrix())
                    non_block_pcds[poseKey] = pcd
                else:
                    print(f"Pose key {poseKey} in block {blockKey} not found in pcds")
    return block_pcds, non_block_pcds

def saveOdomTUM(timestamps: dict[int, float], values: gtsam.Values, filename: str|Path) -> bool:
    filename = Path(filename)
    if not values or not timestamps:
        print("No values or timestamps to save.")
        return False
    
    ret = True
    with open(filename, "w") as f:
        for key in values.keys():
            pose = values.atPose3(key)
            q = pose.rotation().toQuaternion()
            timestamp = timestamps[key]
            if timestamp is None:
                print(f"Warning: Timestamp for key {key} is None. Skipping.")
                continue
            if f.write(f"{timestamp:.6f} {pose.x():.6f} {pose.y():.6f} {pose.z():.6f} {q.x():.6f} {q.y():.6f} {q.z():.6f} {q.w():.6f}\n") == 0:
                print(f"Failed to write pose for key {key} to {filename}")
                ret = False
    return ret

def saveTimestampsWithKeys(timestamp_dict: dict[int, float], filename: str|Path) -> bool:
    filename = Path(filename)
    if not timestamp_dict:
        print("No timestamps to save.")
        return False
    ret = True
    with open(filename, "w") as f:
        for key, timestamp in timestamp_dict.items():
            if f.write(f"{key} {timestamp:.6f}\n") == 0:
                print(f"Failed to write timestamp for key {key} to {filename}")
                ret = False
    return ret

def loadTimestampsWithKeys(filename: str|Path) -> dict[int, float]:
    filename = Path(filename)
    timestamp_dict = {}
    with open(filename, "r") as f:
        for line in f:
            key_str, timestamp_str = line.strip().split()
            key = int(key_str)
            timestamp = float(timestamp_str)
            timestamp_dict[key] = timestamp
    return timestamp_dict

def savePointClouds(pcds: dict[Any, o3d.geometry.PointCloud], 
                    output_dir: str, 
                    remap_keys: dict[Any, Any] | None = None, 
                    remove_existing: bool = True) -> bool:
    
    if pcds is None or len(pcds) == 0:
        print("No point clouds to save.")
        return False

    if remove_existing and os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    ret = True
    for key, pcd in pcds.items():
        if remap_keys:
            key = remap_keys[key]

        filename = os.path.join(output_dir, f"{key}.pcd")
        if not o3d.io.write_point_cloud(filename, pcd):
            print(f"Failed to save point cloud for key {key} to {filename}")
            ret = False

    return ret

def savePointClouds_BlockMap(block_pcds: dict[str, o3d.geometry.PointCloud], 
                             non_block_pcds: dict[int | str, o3d.geometry.PointCloud],
                             output_dir: str, 
                             pose2block_mapping: dict[int, str] | None = None, 
                             remap_keys: dict[int, int] | None = None, 
                             remove_existing: bool = True) -> bool:
    
    blockDir = os.path.join(output_dir, "blocks")
    nonBlockDir = os.path.join(output_dir, "non_blocks")

    if remove_existing and os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    if pose2block_mapping is not None:
        pose2block_file = os.path.join(output_dir, "pose2block_mapping.txt")
        with open(pose2block_file, "w") as f:
            for pose_key, block_key in pose2block_mapping.items():
                f.write(f"{pose_key} {block_key}\n")
    
    ret = True
    if not savePointClouds(block_pcds, blockDir, remap_keys=None, remove_existing=False):
        print("Failed to save block point clouds.")
        ret = False
    if not savePointClouds(non_block_pcds, nonBlockDir, remap_keys=remap_keys, remove_existing=False):
        print("Failed to save non-block point clouds.")
        ret = False

    return ret


def loadPose2BlockMapping(filename: str, reverse: bool = False) -> dict[int, str] | dict[str, int]:
    """
    Load the mapping from pose keys to block keys from a text file.
    The text file should have lines in the format "pose_key: block_key".
    :param filename: Name of the text file containing the mapping
    :param reverse: If True, the returned mapping will be from block keys to pose keys
    :return: A dictionary mapping pose keys to block keys (or block keys to pose keys if reverse is True)
    """

    pose2block_mapping = {}
    with open(filename, "r") as f:
        for line in f:
            if " " in line:
                pose_key_str, block_key = line.strip().split(" ")
                pose_key = int(pose_key_str)
                pose2block_mapping[pose_key] = block_key
    if reverse:
        block2pose_mapping = {v: k for k, v in pose2block_mapping.items()}
        return block2pose_mapping
    return pose2block_mapping


def cleanFactorGraph(graph: gtsam.gtsam.NonlinearFactorGraph|gtsam.ISAM2|gtsam.IncrementalFixedLagSmoother, excludedSymbols:list[str]=[]) -> gtsam.gtsam.NonlinearFactorGraph:
    '''
    Factor graphs from ISAM2 can contain null pointers due to marginalization.
    This function clears out the null pointers and returns a clean graph.
    
    :param graph: gtsam NonlinearFactorGraph or isam2 variable
    :param excludedSymbols: list of single character strings. If any of the factor keys are found in this list, the factor is not added to the graph.
    '''

    if isinstance(graph, gtsam.ISAM2):
        graph = graph.getFactorsUnsafe()
    elif isinstance(graph, gtsam.IncrementalFixedLagSmoother):
        graph = graph.getFactors()

    compact_graph = gtsam.NonlinearFactorGraph()

    for i in range(graph.size()):

        factor = graph.at(i)

        if factor is not None:
            add = True
            for k in factor.keys():
                if chr(gtsam.Symbol(k).chr()) in excludedSymbols:
                    add = False
            if add:
                compact_graph.push_back(factor)
    
    return compact_graph



def GetPointsFromGraphEstimate(values: gtsam.Values, 
                               symbolChr:str|list[str]|None=None, 
                               allOthers:bool=False, 
                               keys:list[int]|set[int]=[],
                               keysToSkip:list[int]|set[int]=[]
                               ) -> tuple[list[list[float]], list[int]]:

    points_list = []
    key_list = []

    if not keys:
        keys = values.keys()

    for key in keys:
        if key in keysToSkip:
            continue
        if symbolChr is not None:
            s = gtsam.Symbol(key)
            if allOthers:
                # if s.chr():
                if chr(s.chr()) in symbolChr:
                    continue
                # elif '' in symbolChr: # The symbol character is empty so if symbolChr is not empty, continue
                #     continue
            else:
                if chr(s.chr()) not in symbolChr: # If the symbol character doesn't match, continue
                    continue
                # elif '' not in symbolChr: # The symbol character is empty so if symbolChr is not empty, continue
                #     continue
        point = values.atPose3(key)

        points_list.append([point.x(), point.y(), point.z()])
        key_list.append(key)

    return points_list, key_list

def GetPosesFromGraphEstimate(values: gtsam.Values) -> tuple[np.ndarray, list[int]]:

    pose_list = []
    key_list = []

    for key in values.keys():
        point = values.atPose3(key)

        quat = point.rotation().toQuaternion()
        pose_list.append([point.x(), point.y(), point.z(), quat.w(), quat.x(), quat.y(), quat.z()])
        key_list.append(key)

    return np.asarray(pose_list), key_list


def createGlobalPCD(
    pcds: dict[int, o3d.geometry.PointCloud],
    values: gtsam.Values,
    keys: list[int] | None = None,
    color: tuple[float, float, float] | None = None,
    voxel_size: float | None = None,
) -> o3d.geometry.PointCloud:
    '''
    Merge per-pose point clouds into a single world-frame point cloud.

    :param pcds: Dict mapping key -> local-frame PointCloud.
    :param values: GTSAM Values containing Pose3 estimates keyed by the same keys.
    :param keys: Subset of keys to merge. Defaults to all keys in `values`.
    :param color: Optional uniform RGB color applied after merging, e.g. (0.6, 0.6, 0.6).
    :param voxel_size: If provided, voxel-downsample the merged cloud before returning.
    '''
    if keys is None:
        keys = list(values.keys())

    global_pcd = o3d.geometry.PointCloud()
    for key in keys:
        if key not in pcds:
            continue
        pcd = copy.deepcopy(pcds[key])
        pcd.transform(values.atPose3(key).matrix())
        global_pcd += pcd

    if color is not None:
        global_pcd.paint_uniform_color(color)
    if voxel_size is not None:
        global_pcd = global_pcd.voxel_down_sample(voxel_size)

    return global_pcd


def getPriorKeys(values: gtsam.Values, currentSessionSymbol: str) -> list[int]:
    '''Return all keys in `values` that do not belong to the current session symbol.'''
    return [k for k in values.keys() if chr(gtsam.Symbol(k).chr()) != currentSessionSymbol]


def getFixedLagKeys(sessionKeys: dict[str, list[int]], currentSessionSymbol: str) -> list[int]:
    '''Return the sliding-window keys for the current session (copy).'''
    return list(sessionKeys.get(currentSessionSymbol, []))


def getCurrentScanKey(sessionKeys: dict[str, list[int]], currentSessionSymbol: str) -> int | None:
    '''Return the most recent key in the fixed-lag window, or None if empty.'''
    keys = sessionKeys.get(currentSessionSymbol, [])
    return keys[-1] if keys else None


# Copied from https://www.open3d.org/docs/0.19.0/tutorial/pipelines/icp_registration.html as it is given as part of their example
def draw_registration_result(source, target, transformation, title:str=""):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                    #   zoom=0.4459,
                                    #   front=[0.9288, -0.2951, -0.2242],
                                    #   lookat=[1.6784, 2.0612, 1.4451],
                                    #   up=[-0.3402, -0.9189, -0.1996],
                                      window_name=title)


def ScanMatch(
        source: o3d.geometry.PointCloud, 
        target: o3d.geometry.PointCloud, 
        trans_init: np.ndarray = np.eye(4), 
        threshold: float = 0.02, 
        showResult: bool = False,
        voxel_size: float | None = 0.5,
        ) -> o3d.pipelines.registration.RegistrationResult:
    '''
    Inputs:
    :param source: transform applies to this one
    :param target: stationary/reference
    :param voxel_size: If not None, will voxel downsample the point clouds before registration. Default=0.5
    '''

    source = copy.deepcopy(source)
    target = copy.deepcopy(target)

    if voxel_size is not None:
        # source = source.voxel_down_sample(voxel_size)
        target = target.voxel_down_sample(voxel_size)
    
    # if showResult:
    #     print("Initial alignment")
    #     evaluation = o3d.pipelines.registration.evaluate_registration(
    #         source, target, threshold, trans_init
    #     )
    #     print(evaluation)
    #     print("Initial Transformation is:")
    #     print(trans_init)
    #     draw_registration_result(source, target, trans_init, title="Initial Transform")

    reg = o3d.pipelines.registration.registration_generalized_icp(
        source, target, 0.5, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    )
    
    trans_final = reg.transformation
    if showResult:
        print("Apply GICP ICP")
        print(reg)
        print("Transformation is:")
        print(trans_final)
        draw_registration_result(source, target, trans_final, title="Final Transform")
    
    return reg

def ListToGtsamKeyList(listToConvert:list) -> gtsam.KeyList:
    keyList = gtsam.KeyList()
    for k in listToConvert:
        keyList.push_back(k)
    return keyList

def AddValueToDictList(dictionary: dict[Any, list], key: Any, value: Any):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]



    
    
import io
import graphviz
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def show_graphviz(dot_string):
    # # 1. Generate dot data
    # dot_string = "digraph G { A -> B; }"
    # plt.close("graphViz")
    plt.figure("graphViz")

    src = graphviz.Source(dot_string)

    # 2. Pipe to PNG bytes in memory
    png_bytes = src.pipe(format="png")
    fp = io.BytesIO(png_bytes)

    # 3. Plot via Matplotlib
    img = mpimg.imread(fp, format="png")
    plt.imshow(img)
    plt.axis("off")  # Hide grid lines/axes
    plt.pause(0.1)   # Opens a live window; script pauses until closed