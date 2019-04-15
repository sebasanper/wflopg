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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import xarray as xr


def draw_windrose(axes, wind_direction_pmf, color='b'):
    """Draw a windrose.

    Parameters
    ----------
    axes
        matplotlib `axes` object with `projection='polar'`
    wind_direction_pmf
        xarray `DataArray` object with direction dimension.
        No normalization is applied to the `DataArray` values
    color
        matplotlib color specification

    """
    axes.set_aspect(1.0)
    axes.set_theta_zero_location("N")
    axes.set_theta_direction(-1)
    axes.set_ylim(0, 1.1 * wind_direction_pmf.max().values.item())
    axes.bar(
        wind_direction_pmf.direction / 360 * 2 * np.pi, wind_direction_pmf,
        width=2 * np.pi / len(wind_direction_pmf),
        color=color
    )


def site_setup(axes):
    """Setup the axes for site plots.

    Parameters
    ----------
    axes
        matplotlib `axes` object

    """
    axes.set_aspect('equal')
    axes.set_axis_off()
    axes.set_xlim(-1.01, 1.01)
    axes.set_ylim(-1.01, 1.01)


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


def draw_convergence(axes, history, max_length=None,
                     min_loss_percentage=0, max_loss_percentage=None):
    """Draw a convergence plot for an optimization run.

    Parameters
    ----------
    axes
        matplotlib `axes` object
    history
        sequence of xarray `Dataset` objects with direction dimension
        No normalization is applied to the `DataArray` values

    """
    axes.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
    max_length = len(history) if max_length is None else max_length
    loss_percentage = 100 * np.array([ds.objective for ds in history])
    if max_loss_percentage is None:
        max_loss_percentage = loss_percentage.max()
    axes.set_xlim(1, 1+max_length)
    axes.set_ylim(min_loss_percentage, max_loss_percentage)
    axes.semilogx(np.arange(1, 1+len(history)), loss_percentage)


def draw_scaling(axes, history, max_length=None):
    """Draw a relative scale factor plot.

    Parameters
    ----------
    axes
        matplotlib `axes` object
    history
        sequence of xarray `Dataset` objects with direction dimension
        No normalization is applied to the `DataArray` values

    """
    axes.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
    max_length = len(history) if max_length is None else max_length
    scales = np.array([ds.scale for ds in history])
    if 'method' in history[-1].attrs:
        methods = np.array([ds.method for ds in history])
    down = methods == 'down'
    axes.semilogy(np.flatnonzero(down), scales[down], '>')
    back = methods == 'back'
    axes.semilogy(np.flatnonzero(back), scales[back], '<')
    cross = methods == 'cross'
    axes.semilogy(np.flatnonzero(cross), scales[cross], 'X')
