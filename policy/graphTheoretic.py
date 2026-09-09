from policy.base import NodeDecisionPolicy
from metrics import Metrics
import numpy as np
from params import Params
from common.customTypes import NodeDecision

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class TwoStepConnectivityPolicy(NodeDecisionPolicy):
    """Map/localize switching from Montano-Oliván et al., arXiv:2602.17226 §III-B2.

    1. Track the T-weighted average degree d-bar against its moving average (mu, sigma).
    2. If d-bar drops more than 2 sigma, evaluate the E-weighted Fiedler value lambda_2.
       - lambda_2 == 0: graphs are disconnected -> MAP
       - lambda_2 > 0 but d-bar dropped more than 4 sigma: weakly connected -> MAP
    3. While mapping, switch back to LOCALIZE only when d-bar is back within 2 sigma
       of the (frozen) localization baseline AND lambda_2 > 0.
    """

    # Algebraic connectivity is identically 0 if the graph is disconnected.
    LAMBDA2_EPS = 1e-8

    def __init__(self, params: Params):
        super().__init__(params)
        self.d_bar_hist = []
        self.d_bar_hist_current = []
        self.debug = params.twoStepConnectivityPolicy.debug
        if params.twoStepConnectivityPolicy.showDLplot:
            _, self.ax = plt.subplots()
            _, self.ax2 = plt.subplots()
        self.previousState_ = NodeDecision.INITIALIZE

    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if not hasattr(metrics, "wpg"):
            raise AttributeError("TwoStepConnectivityPolicy: metrics doesn't have the needed WeightedPoseGraph object")

        d_bar = metrics.d_bar
        current_state = metrics.currentState
        cfg = self.params.twoStepConnectivityPolicy
        window = cfg.movingAverageLength

        # Build the localization baseline before making any MAP/LOCALIZE decision.
        # Samples collected while mapping would pull mu down and collapse the test.
        if len(self.d_bar_hist) < window:
            # if current_state != NodeDecision.MAP:
            self.d_bar_hist.append(d_bar)
            # return NodeDecision.LOCALIZE
            if self.previousState_ == NodeDecision.INITIALIZE:
                lambda2 = metrics.ComputeFiedler()
                if (lambda2 > self.LAMBDA2_EPS):
                    self.previousState_ = NodeDecision.LOCALIZE
                else:
                    self.previousState_ = NodeDecision.MAP
                if self.debug:
                    print(f"TWO_STEP_VALIDATION - Starting in {self.previousState_.name} mode")
            return self.previousState_


        mu = float(np.mean(self.d_bar_hist))
        sigma = float(np.std(self.d_bar_hist, ddof=1)) if len(self.d_bar_hist) > 1 else 0.0
        # sigma = max(sigma, abs(mu) * cfg.minRelativeSigma, 1e-12)
        sigma = max(sigma, cfg.minRelativeSigma, 1e-12)
        lower_2 = mu - 2.0 * sigma
        lower_4 = mu - 4.0 * sigma

        # lambda_2 is the expensive step; the paper only computes it after the 2-sigma
        # gate, or while mapping so the recovery test can run.
        need_lambda = (
            d_bar < lower_2
            or current_state == NodeDecision.MAP
            or cfg.showDLplot
            or self.debug
        )
        lambda2 = metrics.ComputeFiedler() if need_lambda else np.nan
        connected = bool(need_lambda) and (lambda2 > self.LAMBDA2_EPS)

        if current_state == NodeDecision.MAP:
            # Two-step recovery: d-bar back inside the 2-sigma band AND lambda_2 > 0.
            recovered = (d_bar >= lower_2) and connected
            ret = NodeDecision.LOCALIZE if recovered else NodeDecision.MAP
        else:
            ret = NodeDecision.LOCALIZE
            if d_bar < lower_2:
                if (not connected) or (d_bar < lower_4):
                    ret = NodeDecision.MAP
            # if ret != NodeDecision.MAP:
            if True: # TODO: Debugging.
                self.d_bar_hist.append(d_bar)
                while len(self.d_bar_hist) > window:
                    self.d_bar_hist.pop(0)

        if self.debug:
            print(f"\nd_bar: {d_bar}, mu: {mu}, sigma: {sigma}, lambda_2: {lambda2}")
            print(f"  Step 1: d_bar < mu-2sigma = {d_bar < lower_2}")
            print(f"  Step 2: d_bar < mu-4sigma = {d_bar < lower_4}")
            print(f"          lambda_2 == 0       {need_lambda and not connected}")
            print(f"  decision: {ret.name}\n")

        if cfg.showDLplot:
            plot_lambda = float(lambda2) if need_lambda else 0.0
            plot_d_and_lambda(d_bar, 
                            #   float(lower_2), 
                            #   float(lower_4), 
                              float(2*sigma), 
                              float(4*sigma), 
                              plot_lambda, 
                              ax=self.ax, 
                              ax2=self.ax2, 
                              graphWindowLength=cfg.graphWindowLength
                              )

        return NodeDecision.LOCALIZE # TODO: Remove this
        return ret


def plot_d_and_lambda(d_hist: float|list[float],
                      d2_hist: float|list[float],
                      d4_hist: float|list[float],
                      lambda_hist: float|list[float],
                      ax:Axes|None=None,
                      ax2:Axes|None=None,
                      graphWindowLength: int=15,
                      ) -> Axes:
    def _as_float_list(values: float|list[float]) -> list[float]:
        # Accept either one new sample or a batch of samples to append.
        if np.isscalar(values):
            return [float(values)]
        return [float(value) for value in values]

    if ax is None:
        fig, ax = plt.subplots()
    
    if ax2 is None:
        fig2, ax2 = plt.subplots()

    state = getattr(ax, "_d_and_lambda_plot_state", None)
    if state is None:
        # Create the right-side axis once and keep both line artists for fast updates.
        lambda_ax = ax.twinx()
        d_line, = ax.plot([], [], color="tab:blue", label="d_hist")
        d2_line, = ax2.plot([], [], color="tab:green", label="d2_hist")
        d4_line, = ax2.plot([], [], color="tab:orange", label="d4_hist")
        lambda_line, = lambda_ax.plot([], [], color="tab:red", label="lambda_hist")

        # Label and color each axis so the two scales are easy to read.
        ax.set_xlabel("Sample")
        ax.set_ylabel("d_hist", color="tab:blue")
        lambda_ax.set_ylabel("lambda_hist", color="tab:red")
        ax.tick_params(axis="y", colors="tab:blue")
        lambda_ax.tick_params(axis="y", colors="tab:red")

        state = {
            "d_ax": ax,
            "d2_ax": ax2,
            "d4_ax": ax2,
            "lambda_ax": lambda_ax,
            "d_line": d_line,
            "d2_line": d2_line,
            "d4_line": d4_line,
            "lambda_line": lambda_line,
            "d_x": [],
            "d_y": [],
            "d2_x": [],
            "d2_y": [],
            "d4_x": [],
            "d4_y": [],
            "lambda_x": [],
            "lambda_y": [],
        }
        setattr(ax, "_d_and_lambda_plot_state", state)
        setattr(ax2, "_d_and_lambda_plot_state", state)

    d_values = _as_float_list(d_hist)
    d2_values = _as_float_list(d2_hist)
    d4_values = _as_float_list(d4_hist)
    lambda_values = _as_float_list(lambda_hist)

    # Append only the new samples so repeated calls stay timely for live plotting.
    d_start = len(state["d_y"])
    state["d_x"].extend(range(d_start, d_start + len(d_values)))
    state["d_y"].extend(d_values)

    d2_start = int(state["d2_x"][-1]+1) if len(state["d2_x"]) > 0 else 0
    state["d2_x"].extend(range(d2_start, d2_start + len(d2_values)))
    state["d2_y"].extend(d2_values)
    # if d2_start > graphWindowLength:
    #     state["d2_x"] = state["d2_x"][-graphWindowLength:]
    #     state["d2_y"] = state["d2_y"][-graphWindowLength:]

    d4_start = int(state["d4_x"][-1]+1) if len(state["d4_x"]) > 0 else 0
    state["d4_x"].extend(range(d4_start, d4_start + len(d4_values)))
    state["d4_y"].extend(d4_values)
    # if d4_start > graphWindowLength:
    #     state["d4_x"] = state["d4_x"][-graphWindowLength:]
    #     state["d4_y"] = state["d4_y"][-graphWindowLength:]

    lambda_start = len(state["lambda_y"])
    state["lambda_x"].extend(range(lambda_start, lambda_start + len(lambda_values)))
    state["lambda_y"].extend(lambda_values)

    state["d_line"].set_data(state["d_x"], state["d_y"])
    state["d2_line"].set_data(state["d2_x"], state["d2_y"])
    state["d4_line"].set_data(state["d4_x"], state["d4_y"])
    state["lambda_line"].set_data(state["lambda_x"], state["lambda_y"])

    # Rescale the axes around the updated line data without recreating the plot.
    ax.relim()
    ax.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    state["lambda_ax"].relim()
    state["lambda_ax"].autoscale_view()
    ax.figure.canvas.draw_idle()
    ax2.figure.canvas.draw_idle()
    return ax
