"""Visualization functions

Example usage (given some problem object `o`):

    import wflopg.visualization as vis
    fig, ax = plt.subplots()
    vis.draw_boundaries(ax, o.site_boundaries)
    vis.draw_zones(ax, o.site_parcels)
    vis.draw_turbines(ax, o.rotor_radius_adim)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

"""

import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.ticker as _tkr


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
    axes.set_ylim(0, 1.1 * wind_direction_pmf.max().item())
    axes.bar(
        wind_direction_pmf.direction / 360 * 2 * _np.pi, wind_direction_pmf,
        width=_np.minimum(5 * _np.pi / 180,
                          2 * _np.pi / len(wind_direction_pmf)),
        color=color
    )


def site_setup(axes, extra_space=0.01):
    """Setup the axes for site plots.

    Parameters
    ----------
    axes
        matplotlib `axes` object

    """
    axes.set_aspect('equal')
    axes.set_axis_off()
    axes.set_xlim(-1 - extra_space, 1 + extra_space)
    axes.set_ylim(-1 - extra_space, 1 + extra_space)


def draw_boundaries(axes, boundaries):
    """Draw the boundaries of a site

    Parameters
    ----------
    axes
        matplotlib `axes` object
    boundaries
        the applicable Owflop.site_boundaries attribute

    """
    def draw_boundary(boundary):
        if 'polygon' in boundary:
            axes.add_patch(
                _plt.Polygon(boundary['polygon'], edgecolor='k', fill=False))
        if 'circle' in boundary:
            axes.add_patch(
                _plt.Circle(boundary['circle'], boundary['circle'].radius,
                            edgecolor='k', fill=False))
        if 'exclusions' in boundary:
            for exclusion_boundary in boundary['exclusions']:
                draw_boundary(exclusion_boundary)

    for boundary in boundaries:
        draw_boundary(boundary)


def draw_zones(axes, parcels):
    """Draw the parcels of a site, i.e., all enclaves and exclaves

    Parameters
    ----------
    axes
        matplotlib `axes` object
    parcels
        the applicable Owflop.site_parcels attribute

    """
    def draw_zone(zone, exclusion=True):
        c = 'r' if exclusion else 'b'
        l = '--' if exclusion else '-'
        if 'vertices' in zone:
            axes.add_patch(_plt.Polygon(zone['vertices'],
                                        edgecolor=c, linestyle=l, fill=False))
        if 'circle' in zone:
            axes.add_patch(_plt.Circle(zone['circle'], zone['circle'].radius,
                                       edgecolor=c, linestyle=l, fill=False))
        if 'exclusions' in zone:
            for exclusion_zone in zone['exclusions']:
                draw_zone(exclusion_zone, not exclusion)

    draw_zone(parcels)


def draw_turbines(axes, layout, turbine_size,
                  minimal_proximity=0, inside=None, turbine_color='k'):
    """Draw the turbines and their proximity exclusion zones

    Parameters
    ----------
    axes
        matplotlib `axes` object
    layout
        a farm layout, i.e., an xarray `DataArray` with an `'xy'` dimension and
        one non-`'xy'`-dimension
    turbine_size
        the adimensional rotor radius
    proximity
        the minimal proximity defining the rotor distance constraint
        (0 to ignore)
    inside
        the applicable Owflop.inside function (`None` to ignore)

    """
    if inside is not None:
        in_site = inside(layout)['in_site']
    for (i, position) in enumerate(layout.values):
        if inside is not None:
            turbine_color = 'b' if in_site.values[i] else 'r'
        if minimal_proximity > 0:
            axes.add_patch(
                _plt.Circle(position, minimal_proximity / 2,
                            color='r', linestyle=':', fill=False))
        axes.add_patch(
            _plt.Circle(position, turbine_size, color=turbine_color))


def draw_step(axes, layout, step, turbine_size, scale=1):
    """Draw a layout change step using vectors

    Parameters
    ----------
    axes
        matplotlib `axes` object
    layout
        a farm layout, i.e., an xarray `DataArray` with an `'xy'` dimension and
        one non-`'xy'`-dimension
    step
        a layout change step; effectively a difference of two layouts
    turbine_size
        the adimensional rotor radius
    scale
        scaling of the vectors

    """
    axes.quiver(layout.sel(xy='x'), layout.sel(xy='y'),
                step.sel(xy='x'), step.sel(xy='y'),
                angles='xy', scale_units='xy', scale=scale, width=turbine_size/5,
                zorder=2)


def connect_layouts(axes, layouts):
    """Draw lines between corresponding turbines of an iterator of layouts

    Parameters
    ----------
    axes
        matplotlib `axes` object
    layout
        an xarray `DataArray` of farm layouts, i.e., with an `'iteration'`
        dimension`, an 'xy'` dimension, and one further-dimension identifying
        the turbines

    """
    axes.plot(
        layouts.sel(xy='x', drop=True).transpose(),
        layouts.sel(xy='y', drop=True).transpose(),
        '-k'
    )


def draw_convergence(axes, history):
    """Draw a convergence plot for an optimization run.

    Parameters
    ----------
    axes
        matplotlib `axes` object
    history
        an xarray `Dataset` with `'iteration'` coordinate and `'objective'` and
        `'objective_bound'` `DataArray`s

    """
    axes.xaxis.set_major_locator(_tkr.MaxNLocator(integer=True))
    (100 * history.objective).plot.line(ax=axes)
    (100 * history.objective_bound).plot.line('--', c='gray', ax=axes)
    axes.set_ylabel('')


def draw_step_size(axes, history):
    """Draw a relative scale factor plot.

    Parameters
    ----------
    axes
        matplotlib `axes` object
    history
        an xarray `Dataset` with `'iteration'` coordinate and `'max_step'`,
        `'actual_step'`, `'spread'`, and `'method'` `DataArray`s

    """
    axes.xaxis.set_major_locator(_tkr.MaxNLocator(integer=True))
    kwargs = {'ax': axes, 'yscale': 'log'}
    history.max_step.where(history.method == 'a').plot.line('>', **kwargs)
    history.max_step.where(history.method == 'b').plot.line('<', **kwargs)
    history.max_step.where(history.method == 'c').plot.line('X', **kwargs)
    history.actual_step.plot.line('.', c='gray', **kwargs)
    history.spread.plot.line('--', c='gray', **kwargs)
    axes.set_ylabel('')
