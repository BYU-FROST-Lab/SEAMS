from policy.base import NodeDecisionPolicy
from metrics import Metrics
import numpy as np
from params import Params
from common.weighted_pose_graph_class import WeightedPoseGraph
from common.customTypes import NodeDecision

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class TwoStepConnectivityPolicy(NodeDecisionPolicy):
    def __init__(self, params: Params):
        super().__init__(params)
        self.d_bar_hist = []
        self.lastState = NodeDecision.INITIALIZE
        self.debug = params.twoStepConnectivityPolicy.debug
        if params.twoStepConnectivityPolicy.showDLplot:
            _, self.ax = plt.subplots()
        
        self.previousState_ = NodeDecision.INITIALIZE

    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if not hasattr(metrics, "wpg"):
            raise AttributeError("TwoStepConnectivityPolicy: metrics doesn't have the needed WeightedPoseGraph object")
        
        L_t = metrics.wpg.compute_L(weight='weight_t')
        d_bar = np.trace(L_t) / L_t.shape[0]

        if self.previousState_ == NodeDecision.MAP and metrics.currentState != NodeDecision.MAP:
            self.d_bar_hist.clear()
        self.previousState_ = metrics.currentState

        if len(self.d_bar_hist) < (self.params.twoStepConnectivityPolicy.movingAverageLength // 2):
        # if len(self.d_bar_hist) < 2:
            self.d_bar_hist.append(d_bar)
            return NodeDecision.INITIALIZE
        
        mu = np.mean(self.d_bar_hist)
        sigma = np.std(self.d_bar_hist)
        lambda2 = metrics.wpg.compute_fiedler_value(weight='weight_e')

        if self.debug:
            print(f"d_bar: {d_bar}, mu: {mu}, sigma: {sigma}, lambda_2: {lambda2}")
            print(f"  Step 1: d_bar < mu-2sigma = {d_bar < (mu - 2*sigma)}")
            print(f"  Step 2: d_bar < mu-4sigma = {d_bar < (mu - 4*sigma)}")
            print(f"          lambda_2 = 0        {np.isclose(lambda2, 0)}")

        ret = NodeDecision.LOCALIZE

        if d_bar < (mu - 2*sigma):

            if np.isclose(lambda2, 0) or d_bar < (mu - 4*sigma):
                ret = NodeDecision.MAP

        # if metrics.currentState != NodeDecision.MAP:
        self.d_bar_hist.append(d_bar)
        while len(self.d_bar_hist) > self.params.twoStepConnectivityPolicy.movingAverageLength:
            self.d_bar_hist.pop(0)

        if self.params.twoStepConnectivityPolicy.showDLplot:
            plot_d_and_lambda(d_bar, float(mu - 2*sigma), lambda2, ax=self.ax)

        # return NodeDecision.LOCALIZE # Debugging purposes when don't want to switch to mapping mode
        return ret


def plot_d_and_lambda(d_hist: float|list[float], 
                      d2_hist: float|list[float],
                      lambda_hist: float|list[float],
                      ax:Axes|None=None, 
                      ) -> Axes:
    def _as_float_list(values: float|list[float]) -> list[float]:
        # Accept either one new sample or a batch of samples to append.
        if np.isscalar(values):
            return [float(values)]
        return [float(value) for value in values]

    if ax is None:
        fig, ax = plt.subplots()
    
    state = getattr(ax, "_d_and_lambda_plot_state", None)
    if state is None:
        # Create the right-side axis once and keep both line artists for fast updates.
        lambda_ax = ax.twinx()
        d_line, = ax.plot([], [], color="tab:blue", label="d_hist")
        d2_line, = ax.plot([], [], color="tab:green", label="d2_hist")
        lambda_line, = lambda_ax.plot([], [], color="tab:red", label="lambda_hist")

        # Label and color each axis so the two scales are easy to read.
        ax.set_xlabel("Sample")
        ax.set_ylabel("d_hist", color="tab:blue")
        lambda_ax.set_ylabel("lambda_hist", color="tab:red")
        ax.tick_params(axis="y", colors="tab:blue")
        lambda_ax.tick_params(axis="y", colors="tab:red")

        state = {
            "d_ax": ax,
            "d2_ax": ax,
            "lambda_ax": lambda_ax,
            "d_line": d_line,
            "d2_line": d2_line,
            "lambda_line": lambda_line,
            "d_x": [],
            "d_y": [],
            "d2_x": [],
            "d2_y": [],
            "lambda_x": [],
            "lambda_y": [],
        }
        setattr(ax, "_d_and_lambda_plot_state", state)

    d_values = _as_float_list(d_hist)
    d2_values = _as_float_list(d2_hist)
    lambda_values = _as_float_list(lambda_hist)

    # Append only the new samples so repeated calls stay timely for live plotting.
    d_start = len(state["d_y"])
    state["d_x"].extend(range(d_start, d_start + len(d_values)))
    state["d_y"].extend(d_values)

    d2_start = len(state["d2_y"])
    state["d2_x"].extend(range(d2_start, d2_start + len(d2_values)))
    state["d2_y"].extend(d2_values)

    lambda_start = len(state["lambda_y"])
    state["lambda_x"].extend(range(lambda_start, lambda_start + len(lambda_values)))
    state["lambda_y"].extend(lambda_values)

    state["d_line"].set_data(state["d_x"], state["d_y"])
    state["d2_line"].set_data(state["d2_x"], state["d2_y"])
    state["lambda_line"].set_data(state["lambda_x"], state["lambda_y"])

    # Rescale the axes around the updated line data without recreating the plot.
    ax.relim()
    ax.autoscale_view()
    state["lambda_ax"].relim()
    state["lambda_ax"].autoscale_view()
    ax.figure.canvas.draw_idle()

    

    return ax
