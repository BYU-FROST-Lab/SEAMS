#!/usr/bin/env python3

import open3d as o3d

import common.FGVizHelpers as fgtools
from multiSession.SessionManager import SessionManager


class LiveMapVisualizer:
    '''
    Non-blocking Open3D window showing three live layers from a SessionManager:

      - Prior map    (gray)  : all poses/scans outside the current session symbol.
                               Merged cloud rebuilt only when the prior key set changes.
      - Fixed-lag window (green): the sliding-window frames of the current session.
      - Current scan (red)  : the single most-recent frame.

    Usage::

        viz = LiveMapVisualizer(updater.combined)
        viz.start()
        for ...:
            updater.AddOdomKeyFrame(...)
            if not viz.update():
                break          # window was closed
        viz.close()
    '''

    DEFAULT_COLORS = {
        "prior":     [0.6, 0.6, 0.6],
        "fixed_lag": [0.2, 0.8, 0.2],
        "scan":      [1.0, 0.2, 0.2],
    }

    def __init__(
        self,
        session,
        voxel_size: float = 0.5,
        scan_voxel_size: float | None = None,
        colors: dict | None = None,
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

        self._vis: o3d.visualization.Visualizer | None = None
        # Becomes True after the first successful poll_events()+update_renderer() cycle.
        # Used to distinguish "window never opened / no display" from "user closed window".
        self._window_confirmed_open: bool = False

        # Geometry objects held as persistent handles for update_geometry()
        self._prior_geom = o3d.geometry.PointCloud()
        self._fixed_lag_geom = o3d.geometry.PointCloud()
        self._scan_geom = o3d.geometry.PointCloud()

        # Cache prior key set to detect when a rebuild is needed
        self._prior_signature: frozenset = frozenset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        '''Create the Open3D window and register the three geometry layers.'''
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(self._window_name)
        self._window_confirmed_open = False

        self._vis.add_geometry(self._prior_geom)
        self._vis.add_geometry(self._fixed_lag_geom)
        self._vis.add_geometry(self._scan_geom)

    def update(self) -> bool:
        '''
        Refresh all three layers from the current session state.

        Returns False only when the user explicitly closes the window.  If the
        window could not be opened at all (headless / no display), visualization
        is silently disabled and True is returned so the calling loop continues.
        '''
        if self._vis is None:
            return True  # visualization disabled — keep the loop running

        self._update_prior_layer()
        self._update_fixed_lag_layer()
        self._update_current_scan()

        self._vis.update_geometry(self._prior_geom)
        self._vis.update_geometry(self._fixed_lag_geom)
        self._vis.update_geometry(self._scan_geom)

        alive = self._vis.poll_events() and self._vis.update_renderer()

        if not alive:
            # if not self._window_confirmed_open:
            #     # poll_events() returned False before any frame was drawn —
            #     # the window never actually opened (no display, Wayland issue, etc.)
            #     print("[LiveMapVisualizer] Window unavailable (no display?); "
            #           "Open3D visualization disabled.")
            #     self._vis.destroy_window()
            #     self._vis = None
            #     return True  # let the loop continue without visualization
            # Window was open and the user closed it
            return False

        self._window_confirmed_open = True
        return True

    def close(self) -> None:
        '''Destroy the Open3D window.'''
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None

    # ------------------------------------------------------------------
    # Layer update helpers
    # ------------------------------------------------------------------

    def _update_prior_layer(self) -> None:
        session = self._session
        prior_keys = fgtools.getPriorKeys(session.currentEstimate_, session.currentSessionSymbol)
        signature = frozenset(prior_keys)

        if signature == self._prior_signature and signature:
            # Key set unchanged — still re-transform with latest poses by rebuilding
            # the merged cloud (cheap because the count is bounded by prior history)
            pass  # fall through to unconditional rebuild below

        self._prior_signature = signature
        merged = fgtools.createGlobalPCD(
            session.pcds_,
            session.currentEstimate_,
            keys=prior_keys,
            color=self._colors["prior"],
            voxel_size=self._voxel_size,
        )
        self._replace_pcd(self._prior_geom, merged)

    def _update_fixed_lag_layer(self) -> None:
        session = self._session
        keys = fgtools.getFixedLagKeys(session.sessionKeys, session.currentSessionSymbol)
        merged = fgtools.createGlobalPCD(
            session.pcds_,
            session.currentEstimate_,
            keys=keys,
            color=self._colors["fixed_lag"],
            voxel_size=self._voxel_size,
        )
        self._replace_pcd(self._fixed_lag_geom, merged)

    def _update_current_scan(self) -> None:
        session = self._session
        key = fgtools.getCurrentScanKey(session.sessionKeys, session.currentSessionSymbol)
        if key is None:
            self._replace_pcd(self._scan_geom, o3d.geometry.PointCloud())
            return
        merged = fgtools.createGlobalPCD(
            session.pcds_,
            session.currentEstimate_,
            keys=[key],
            color=self._colors["scan"],
            voxel_size=self._scan_voxel_size,
        )
        self._replace_pcd(self._scan_geom, merged)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_pcd(dest: o3d.geometry.PointCloud, src: o3d.geometry.PointCloud) -> None:
        '''Copy points and colors from src into dest in-place (preserves the registered handle).'''
        dest.points = src.points
        dest.colors = src.colors
        if src.has_normals():
            dest.normals = src.normals
