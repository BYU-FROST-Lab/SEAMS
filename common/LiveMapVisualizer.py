#!/usr/bin/env python3
"""
Non-blocking live Open3D map visualizer.

The Open3D event loop runs in a dedicated background thread so that the window
stays responsive regardless of how long the main processing loop takes between
calls to update().  The main thread computes new geometry data and hands it off
to the viz thread via a lock-protected pending buffer.
"""

import threading
import time
from typing import Optional

import open3d as o3d

import common.FGVizHelpers as fgtools
from multiSession.SessionManager import SessionManager


class LiveMapVisualizer:
    '''
    Non-blocking Open3D window showing three live layers from a SessionManager:

      - Prior map    (gray)  : all poses/scans outside the current session symbol.
      - Fixed-lag window (green): the sliding-window frames of the current session.
      - Current scan (red)  : the single most-recent frame.

    The Open3D event loop runs in a background thread.  The window stays alive
    and responsive even when the main thread spends several seconds processing a
    keyframe.

    Usage::

        viz = LiveMapVisualizer(session)
        viz.start()
        for ...:
            session.Update(...)
            if not viz.update():
                break          # user closed the window
        viz.close()
    '''

    DEFAULT_COLORS = {
        "prior":     [0.6, 0.6, 0.6],
        "fixed_lag": [0.2, 0.8, 0.2],
        "scan":      [1.0, 0.2, 0.2],
    }

    # How long to wait for the window to confirm it opened successfully.
    _WINDOW_OPEN_TIMEOUT_S: float = 5.0
    # Target frame rate for the viz thread event loop.
    _TARGET_FPS: float = 30.0

    def __init__(
        self,
        session,
        voxel_size: float = 0.5,
        scan_voxel_size: Optional[float] = None,
        colors: Optional[dict] = None,
        window_name: str = "Live Map",
    ):
        '''
        :param session: A SessionManager (or SessionUpdater — uses .combined if so).
        :param voxel_size: Voxel size used when downsampling the prior and fixed-lag layers.
        :param scan_voxel_size: Voxel size for the current-scan layer (None = full resolution).
        :param colors: Override display colors for "prior", "fixed_lag", and/or "scan".
        :param window_name: Title of the Open3D window.
        '''
        # Accept SessionUpdater by transparently unwrapping .combined
        if hasattr(session, 'combined'):
            session = session.combined
        if not isinstance(session, SessionManager):
            raise TypeError(f"Expected SessionManager, got {type(session)}")

        self._session = session
        self._voxel_size = voxel_size
        self._scan_voxel_size = scan_voxel_size
        self._window_name = window_name
        self._colors = {**self.DEFAULT_COLORS, **(colors or {})}

        # Threading primitives
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # Set by the viz thread once the window has rendered at least one frame.
        self._window_alive = threading.Event()
        # Set to ask the viz thread to exit cleanly.
        self._stop_event = threading.Event()

        # Geometry handles registered with the Visualizer (only touched by viz thread).
        # Created here so __init__ can be called on the main thread before start().
        self._prior_geom = o3d.geometry.PointCloud()
        self._fixed_lag_geom = o3d.geometry.PointCloud()
        self._scan_geom = o3d.geometry.PointCloud()

        # Pending geometry data produced by the main thread and consumed by the viz thread.
        self._pending_prior: Optional[o3d.geometry.PointCloud] = None
        self._pending_fixed_lag: Optional[o3d.geometry.PointCloud] = None
        self._pending_scan: Optional[o3d.geometry.PointCloud] = None
        self._has_pending: bool = False

        # Cache prior key set to skip unnecessary rebuilds.
        self._prior_signature: frozenset = frozenset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        '''
        Launch the background visualization thread.

        Blocks until the window has confirmed it is alive (up to
        _WINDOW_OPEN_TIMEOUT_S seconds).  If the window cannot open (no display,
        headless environment), a warning is printed and the visualizer is silently
        disabled; subsequent calls to update() return True so the calling loop
        continues unaffected.
        '''
        self._stop_event.clear()
        self._window_alive.clear()

        self._thread = threading.Thread(
            target=self._viz_thread,
            name="open3d-viz",
            daemon=True,
        )
        self._thread.start()

        # Wait for the window to confirm it is open.
        if not self._window_alive.wait(timeout=self._WINDOW_OPEN_TIMEOUT_S):
            print("[LiveMapVisualizer] Window failed to open (no display?); "
                  "Open3D visualization disabled.")
            self._stop_event.set()
            self._thread = None

    def update(self) -> bool:
        '''
        Push fresh geometry data to the visualization window.

        Computes the three layers from the current session state on the calling
        (main) thread and hands the result to the viz thread.

        Returns False when the user has closed the Open3D window.  Returns True
        in all other cases (including when visualization is disabled), so the
        calling loop can always check the return value safely.
        '''
        # Viz is disabled (failed to open or already closed).
        if self._thread is None:
            return True

        # Window was closed by the user (viz thread exited).
        if self._stop_event.is_set() or not self._thread.is_alive():
            return False

        # Compute new geometry on the main thread (safe: main thread owns session).
        prior = self._compute_prior_layer()
        fixed_lag = self._compute_fixed_lag_layer()
        scan = self._compute_scan_layer()

        with self._lock:
            self._pending_prior = prior
            self._pending_fixed_lag = fixed_lag
            self._pending_scan = scan
            self._has_pending = True

        return True

    def close(self) -> None:
        '''Signal the viz thread to stop and wait for it to exit.'''
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    # ------------------------------------------------------------------
    # Viz thread
    # ------------------------------------------------------------------

    def _viz_thread(self) -> None:
        '''Main function of the background Open3D thread.'''
        frame_period = 1.0 / self._TARGET_FPS
        vis = o3d.visualization.Visualizer()
        vis.create_window(self._window_name)

        vis.add_geometry(self._prior_geom)
        vis.add_geometry(self._fixed_lag_geom)
        vis.add_geometry(self._scan_geom)

        # Track whether we have received any real data yet so we can
        # reset the camera to fit the geometry on the first update.
        first_real_update = True

        while not self._stop_event.is_set():
            # Apply any pending geometry updates from the main thread.
            geom_updated = False
            with self._lock:
                if self._has_pending:
                    _replace_pcd(self._prior_geom, self._pending_prior)
                    _replace_pcd(self._fixed_lag_geom, self._pending_fixed_lag)
                    _replace_pcd(self._scan_geom, self._pending_scan)
                    self._pending_prior = None
                    self._pending_fixed_lag = None
                    self._pending_scan = None
                    self._has_pending = False
                    vis.update_geometry(self._prior_geom)
                    vis.update_geometry(self._fixed_lag_geom)
                    vis.update_geometry(self._scan_geom)
                    geom_updated = True

            # On the first update that contains real points, reset the camera
            # so it frames the actual geometry rather than the empty initial scene.
            if geom_updated and first_real_update:
                has_points = (
                    self._prior_geom.has_points()
                    or self._fixed_lag_geom.has_points()
                    or self._scan_geom.has_points()
                )
                if has_points:
                    vis.reset_view_point(True)
                    first_real_update = False

            if not vis.poll_events():
                # User closed the window.
                break
            vis.update_renderer()

            # Signal that the window has rendered at least one frame.
            self._window_alive.set()

            time.sleep(frame_period)

        vis.destroy_window()
        # Ensure the stop event is set so update() notices.
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Geometry computation (called on main thread inside update())
    # ------------------------------------------------------------------

    def _compute_prior_layer(self) -> o3d.geometry.PointCloud:
        session = self._session
        prior_keys = fgtools.getPriorKeys(session.currentEstimate_, session.currentSessionSymbol)
        return fgtools.createGlobalPCD(
            session.pcds_,
            session.currentEstimate_,
            keys=prior_keys,
            color=self._colors["prior"],
            voxel_size=self._voxel_size,
        )

    def _compute_fixed_lag_layer(self) -> o3d.geometry.PointCloud:
        session = self._session
        keys = fgtools.getFixedLagKeys(session.sessionKeys, session.currentSessionSymbol)
        return fgtools.createGlobalPCD(
            session.pcds_,
            session.currentEstimate_,
            keys=keys,
            color=self._colors["fixed_lag"],
            voxel_size=self._voxel_size,
        )

    def _compute_scan_layer(self) -> o3d.geometry.PointCloud:
        session = self._session
        key = fgtools.getCurrentScanKey(session.sessionKeys, session.currentSessionSymbol)
        if key is None:
            return o3d.geometry.PointCloud()
        return fgtools.createGlobalPCD(
            session.pcds_,
            session.currentEstimate_,
            keys=[key],
            color=self._colors["scan"],
            voxel_size=self._scan_voxel_size,
        )


# ------------------------------------------------------------------
# Module-level utility (no instance needed)
# ------------------------------------------------------------------

def _replace_pcd(dest: o3d.geometry.PointCloud, src: Optional[o3d.geometry.PointCloud]) -> None:
    '''Copy points, colors, and normals from src into dest in-place.'''
    if src is None:
        dest.points = o3d.utility.Vector3dVector()
        dest.colors = o3d.utility.Vector3dVector()
        dest.normals = o3d.utility.Vector3dVector()
        return
    dest.points = src.points
    dest.colors = src.colors
    if src.has_normals():
        dest.normals = src.normals
    else:
        dest.normals = o3d.utility.Vector3dVector()
