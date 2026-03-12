import numpy as np
import pyvista as pv


def draw_coord_frame(plotter, T, name=None, opacity=1., scale=1.):
    tip_length = 0.25  # * scale
    tip_radius = 0.08 * scale
    tip_resolution = 20
    shaft_radius = 0.04 * scale
    shaft_resolution = 20
    scale = 0.05  # * scale
    if name is None:
        name = str(np.random.uniform())
    plotter.add_mesh(pv.Arrow(T[:3, 3], T[:3, 0],
                              tip_length=tip_length,
                              tip_radius=tip_radius,
                              tip_resolution=tip_resolution,
                              shaft_radius=shaft_radius,
                              shaft_resolution=shaft_resolution,
                              scale=scale), color='r', name=name + 'x', opacity=opacity)
    plotter.add_mesh(pv.Arrow(T[:3, 3], T[:3, 1],
                              tip_length=tip_length,
                              tip_radius=tip_radius,
                              tip_resolution=tip_resolution,
                              shaft_radius=shaft_radius,
                              shaft_resolution=shaft_resolution,
                              scale=scale), color='g', name=name + 'y', opacity=opacity)
    plotter.add_mesh(pv.Arrow(T[:3, 3], T[:3, 2],
                              tip_length=tip_length,
                              tip_radius=tip_radius,
                              tip_resolution=tip_resolution,
                              shaft_radius=shaft_radius,
                              shaft_resolution=shaft_resolution,
                              scale=scale), color='b', name=name + 'z', opacity=opacity)


