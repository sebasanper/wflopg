"""Visualization functions

Example usage (given some problem object `o`):

    import wflopg.visualization as vis
    fig, ax = plt.subplots()
    vis.draw_boundaries(ax, o)
    vis.draw_zones(ax, o)
    vis.draw_turbines(ax, o)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

"""

import matplotlib.pyplot as plt
import xarray as xr


def draw_boundaries(axes, owflop):
    """Draw the boundaries of a site"""
    def draw_boundary(boundary):
        if 'polygon' in boundary:
            axes.add_patch(
                plt.Polygon(boundary['polygon'], edgecolor='k', fill=False))
        if 'circle' in boundary:
            axes.add_patch(
                plt.Circle(boundary['circle'], boundary['circle'].radius,
                           edgecolor='k', fill=False))
        if 'exclusions' in boundary:
            for exclusion_boundary in boundary['exclusions']:
                draw_boundary(exclusion_boundary)

    for boundary in owflop.boundaries:
        draw_boundary(boundary)


def draw_zones(axes, owflop):
    """Draw the parcels of a site, i.e., all enclaves and exclaves"""
    def draw_zone(zone, exclusion=True):
        c = 'r' if exclusion else 'b'
        l = '--' if exclusion else '-'
        if 'vertices' in zone:
            axes.add_patch(plt.Polygon(zone['vertices'],
                                       edgecolor=c, linestyle=l, fill=False))
        if 'circle' in zone:
            axes.add_patch(plt.Circle(zone['circle'], zone['circle'].radius,
                                      edgecolor=c, linestyle=l, fill=False))
        if 'exclusions' in zone:
            for exclusion_zone in zone['exclusions']:
                draw_zone(exclusion_zone, not exclusion)

    draw_zone(owflop.parcels)


def draw_turbines(axes, owflop, layout=None, proximity=False, in_or_out=False):
    """Draw the turbines and their proximity exclusion zones

    The layout is assumed to have only a single non-xy dimension.

    """
    if layout is None:
        layout = owflop.history[-1]['layout']
    turbine_size = owflop.rotor_radius / owflop.site_radius
    turbine_color = 'k'
    if in_or_out:
        inside = owflop.inside(layout)['in_site']
    for (i, position) in enumerate(layout.values):
        if in_or_out:
            turbine_color = 'b' if inside.values[i] else 'r'
        if proximity:
            axes.add_patch(
                plt.Circle(position, owflop.minimal_proximity / 2,
                           color='r', linestyle=':', fill=False))
        axes.add_patch(plt.Circle(position, turbine_size, color=turbine_color))


def draw_step(axes, owflop, layout, step):
    """Draw a layout change step using vectors"""
    turbine_size = owflop.rotor_radius / owflop.site_radius
    axes.quiver(layout.sel(xy='x'), layout.sel(xy='y'),
                step.sel(xy='x'), step.sel(xy='y'),
                angles='xy', scale_units='xy', scale=1, width=turbine_size/2)


def connect_layouts(axes, layouts):
    """Draw lines between corresponding turbines of an iterator of layouts"""
    xs = xr.concat(
        [layout.sel(xy='x', drop=True) for layout in layouts], dim='layout')
    ys = xr.concat(
        [layout.sel(xy='y', drop=True) for layout in layouts], dim='layout')
    axes.plot(xs, ys, '-k')
