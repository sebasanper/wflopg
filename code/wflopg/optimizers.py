import sys as _sys
import numpy as _np
import xarray as _xr
import matplotlib.pyplot as _plt
import matplotlib.gridspec as _gs

import wflopg.visualization as vis
from wflopg.create_layout import _take_step, fix_constraints
from wflopg.helpers import rss


def _iterate(step_generator, owflop, max_iterations, step_normalizer):

    def layout2power(owflop):
        owflop.calculate_geometry()
        owflop.calculate_deficit()
        owflop.calculate_power()

    def update_history(owflop, corrections):
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = owflop._ds.layout
        owflop.history[-1]['objective'] = owflop.objective()
        owflop.history[-1].attrs['corrections'] = corrections

    layout2power(owflop)
    update_history(owflop, '')
    best = start = owflop.history[0].objective
    for iteration in range(1, max_iterations+1):
        print(iteration, end=': ')
        # calculate new layout
        owflop.calculate_relative_wake_loss_vector()
        step = step_generator()
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift # TODO: remove shift before normalization
        step -= step.mean(dim='target')
        step *= step_normalizer
        # take step
        _take_step(owflop, step)
        # fix any constraint violation
        corrections = fix_constraints(owflop)
        # evaluate new layout
        layout2power(owflop)
        update_history(owflop, corrections)
        current = owflop.history[-1].objective
        # check best layout and criteria for early termination
        if current < best:
            best = current
        else:
            if _np.log2(iteration) * (current - best) > (start - best):
                break
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout, dim='xy'
            ) / owflop.rotor_diameter_adim
            if distance_from_previous.max() < .1:
                break


def _adaptive_iterate(step_generator, owflop, max_iterations, step_normalizer,
                      scaler=[.5, 1.1], visualize=False):
    if visualize:
        fig = _plt.figure()
        grid = _gs.GridSpec(3, 5)
        ax_windrose = fig.add_subplot(grid[0, :2], polar=True)
        vis.draw_windrose(ax_windrose, owflop._ds.direction_pmf)
        ax_convergence = fig.add_subplot(grid[1, :2])
        vis.draw_convergence(ax_convergence, owflop.history)
        ax_scaling = fig.add_subplot(grid[2, :2], sharex=ax_convergence)
        ax_layout = fig.add_subplot(grid[:, 2:])
        vis.site_setup(ax_layout)
        vis.draw_turbines(ax_layout, owflop, owflop._ds.layout,
                          proximity=True, in_or_out=True)
        vis.draw_boundaries(ax_layout, owflop)
        grid.tight_layout(fig)
        _plt.pause(.10)
    scale_coord = ('scale', ['-', '+'])
    iterations = 0
    best = last = start = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * _xr.DataArray([1, 1], coords=[scale_coord])
    )
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    scaler = _xr.DataArray(scaler, coords=[scale_coord])
    scaling = _xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (
                last - best
                > (start - best) / _np.log2(len(owflop.history) + 2)
            ):
                break
        print(iterations, end=': ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin()
        # we continue from best layout
        owflop._ds['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
            * _xr.DataArray([1, 1], coords=[scale_coord])
        )
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        # update history
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True))
        owflop.history[-1]['objective'] = objectives.isel(scale=i, drop=True)
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['scale'] = scaling[i].item()
        if visualize:
            ax_convergence.clear()
            vis.draw_convergence(ax_convergence, owflop.history)
            ax_scaling.clear()
            vis.draw_scaling(ax_scaling, owflop.history)
            ax_layout.clear()
            vis.site_setup(ax_layout)
            vis.connect_layouts(ax_layout,
                                [ds.layout for ds in owflop.history])
            vis.draw_turbines(ax_layout, owflop, owflop.history[0].layout)
            vis.draw_turbines(ax_layout, owflop, owflop.history[-1].layout,
                              proximity=True, in_or_out=True)
            vis.draw_boundaries(ax_layout, owflop)
            _plt.pause(0.1)
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0].objective
        else:
            last = owflop.history[-1].objective
            if last < best:
                best = last
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout,
                dim='xy'
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < owflop.rotor_diameter_adim / 10:
                break
        # calculate new layout
        scaling = scaling.isel(scale=i) * scaler
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds.relative_deficit.isel(scale=i))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds.wake_loss_factor.isel(scale=i))
        owflop._ds['unit_vector'] = owflop._ds.unit_vector.isel(scale=i)
        owflop.calculate_relative_wake_loss_vector()
        step = step_generator()
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        step *= step_normalizer
        step = step * scaling  # generate the different step variants
        _take_step(owflop, step)
        corrections = fix_constraints(owflop)
        iterations += 1


def pure_down(owflop, max_iterations=_sys.maxsize,
              scaling=False, scaler=[.5, 1.1], multiplier=3, visualize=False):
    """Optimize the layout using push-down only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_down_vector, owflop, max_iterations,
            owflop.rotor_diameter_adim * multiplier, scaler=scaler,
            visualize=visualize
        )
    else:
        _iterate(
            owflop.calculate_push_down_vector, owflop,
            max_iterations, owflop.rotor_diameter_adim * multiplier
        )


def pure_back(owflop, max_iterations=_sys.maxsize,
              scaling=False, scaler=[.5, 1.1], multiplier=3, visualize=False):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_back_vector, owflop, max_iterations,
            owflop.rotor_diameter_adim * multiplier, scaler=scaler,
            visualize=visualize
        )
    else:
        _iterate(
            owflop.calculate_push_back_vector, owflop,
            max_iterations, owflop.rotor_diameter_adim * multiplier
        )


def mixed_down_and_back(owflop, max_iterations=_sys.maxsize,
                        scaling=False, scaler=[.5, 1.1], multiplier=3,
                        visualize=False):
    """Optimize the layout using a mixture of push-down and push-back

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    def step_generator():
        return (owflop.calculate_push_down_vector()
                + owflop.calculate_push_back_vector()) / 2

    if scaling:
        _adaptive_iterate(
            step_generator, owflop, max_iterations,
            owflop.rotor_diameter_adim * multiplier, scaler=scaler,
            visualize=visualize)
    else:
        _iterate(step_generator, owflop, max_iterations,
                 owflop.rotor_diameter_adim * multiplier)


def pure_cross(owflop, max_iterations=_sys.maxsize, scaling=False,
               scaler=[.5, 1.1], multiplier=3, visualize=False):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_cross_vector, owflop, max_iterations,
            owflop.rotor_diameter_adim * multiplier, scaler=scaler,
            visualize=visualize
        )
    else:
        _iterate(
            owflop.calculate_push_cross_vector, owflop,
            max_iterations, owflop.rotor_diameter_adim * multiplier
        )


def multi_adaptive(owflop, max_iterations=_sys.maxsize,
                   scaler=[.5, 1.1], multiplier=3,
                   only_above_average=False, visualize=False):
    if visualize:
        fig = _plt.figure()
        grid = _gs.GridSpec(3, 5)
        ax_windrose = fig.add_subplot(grid[0, :2], polar=True)
        vis.draw_windrose(ax_windrose, owflop._ds.direction_pmf)
        ax_convergence = fig.add_subplot(grid[1, :2])
        vis.draw_convergence(ax_convergence, owflop.history)
        ax_scaling = fig.add_subplot(grid[2, :2], sharex=ax_convergence)
        ax_layout = fig.add_subplot(grid[:, 2:])
        vis.site_setup(ax_layout)
        vis.draw_turbines(ax_layout, owflop, owflop._ds.layout,
                          proximity=True, in_or_out=True)
        vis.draw_boundaries(ax_layout, owflop)
        grid.tight_layout(fig)
        _plt.pause(.10)
    scale_coord = ('scale', ['-', '+'])
    method_coord = ('method', ['down', 'back', 'cross'])
    iterations = 0
    best = last = start = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * _xr.DataArray(
            [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
    )
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    scaler = _xr.DataArray(scaler, coords=[scale_coord])
    scaling = _xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (
                last - best
                > (start - best) / _np.log2(len(owflop.history) + 2)
            ):
                break
        print(iterations, end=': ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin(dim='scale')
        j = objectives.min(dim='scale').argmin('method').item()
        # we continue from best layout
        owflop._ds['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
            * _xr.DataArray(
                [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
        )
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        # update history
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
        )
        owflop.history[-1]['objective'] = (
            objectives.isel(scale=i, drop=True).isel(method=j, drop=True))
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        owflop.history[-1].attrs['scale'] = (
            scaling.isel(scale=i, drop=True).isel(method=j, drop=True).item())
        if visualize:
            ax_convergence.clear()
            vis.draw_convergence(ax_convergence, owflop.history)
            ax_scaling.clear()
            vis.draw_scaling(ax_scaling, owflop.history)
            ax_layout.clear()
            vis.site_setup(ax_layout)
            vis.connect_layouts(ax_layout,
                                [ds.layout for ds in owflop.history])
            vis.draw_turbines(ax_layout, owflop, owflop.history[0].layout)
            vis.draw_turbines(ax_layout, owflop, owflop.history[-1].layout,
                              proximity=True, in_or_out=True)
            vis.draw_boundaries(ax_layout, owflop)
            _plt.pause(0.1)
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0].objective
        else:
            last = owflop.history[-1].objective
            if last < best:
                best = last
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout,
                dim='xy'
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < owflop.rotor_diameter_adim / 10:
                break
        # calculate new layout
        scaling = scaling.isel(scale=i, drop=True) * scaler
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds.relative_deficit.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds.wake_loss_factor.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['unit_vector'] = (
            owflop._ds.unit_vector.isel(scale=i, drop=True)
                                  .isel(method=j, drop=True))
        owflop.calculate_relative_wake_loss_vector()
        down_step = owflop.calculate_push_down_vector()
        back_step = owflop.calculate_push_back_vector()
        cross_step = owflop.calculate_push_cross_vector()
        # throw steps in one big DataArray
        step = _xr.concat([down_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # only take above average steps
        if only_above_average:
            mean_distance = distance.mean(dim='target')
            step *= (distance > mean_distance)
        # generate the different step variants
        step = step * owflop.rotor_diameter_adim * multiplier * scaling
        # take the step
        _take_step(owflop, step)
        corrections = fix_constraints(owflop)
        iterations += 1


def method_chooser(owflop, max_iterations=_sys.maxsize):
    method_coord = ('method', ['down', 'back', 'cross'])
    iterations = 0
    best = last = start = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * _xr.DataArray([1, 1, 1], coords=[method_coord]))
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (
                last - best
                > (start - best) / _np.log2(len(owflop.history) + 2)
            ):
                break
        print(iterations, end=': ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        j = objectives.argmin('method').values.item()
        # we continue from best layout
        owflop._ds['layout'] = (
            owflop._ds.layout.isel(method=j, drop=True)
            * _xr.DataArray([1, 1, 1], coords=[method_coord])
        )
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        # update history
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(method=j, drop=True))
        owflop.history[-1]['objective'] = objectives.isel(method=j, drop=True)
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0].objective
        else:
            last = owflop.history[-1].objective
            if last < best:
                best = last
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout,
                dim='xy'
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < owflop.rotor_diameter_adim / 10:
                break
        # calculate new layouts
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds.relative_deficit.isel(method=j, drop=True))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds.wake_loss_factor.isel(method=j, drop=True))
        owflop._ds['unit_vector'] = (
            owflop._ds.unit_vector.isel(method=j, drop=True))
        owflop.calculate_relative_wake_loss_vector()
        down_step = owflop.calculate_push_down_vector()
        back_step = owflop.calculate_push_back_vector()
        cross_step = owflop.calculate_push_cross_vector()
        # throw steps in one big DataArray
        step = _xr.concat([down_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # take the step, one rotor diameter for the largest pseudo-gradient
        _take_step(owflop, step * owflop.rotor_diameter_adim)
        corrections = fix_constraints(owflop)
        iterations += 1


def multi_wind_resource(owflop, wind_resources, max_iterations=_sys.maxsize,
                        scaler=[.5, 1.1], multiplier=3):
    scale_coord = ('scale', ['-', '+'])
    method_coord = ('method', ['down', 'back', 'cross'])
    # save initial wind resource for objective evaluation
    wind_resource = _xr.Dataset()
    wind_resource['direction_pmf'] = owflop._ds.direction_pmf
    wind_resource['wind_speed_cpmf'] = owflop._ds.wind_speed_cpmf
    iterations = 0
    best = last = start = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * _xr.DataArray(
            [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
    )
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    scaler = _xr.DataArray(scaler, coords=[scale_coord])
    scaling = _xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (
                last - best
                > (start - best) / _np.log2(len(owflop.history) + 2)
            ):
                break
        print(iterations, end=': ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin(dim='scale')
        j = objectives.min(dim='scale').argmin('method').values.item()
        # we continue from best layout
        owflop._ds['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
            * _xr.DataArray(
                [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
        )
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        # update history
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
        )
        owflop.history[-1]['objective'] = (
            objectives.isel(scale=i, drop=True).isel(method=j, drop=True))
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        owflop.history[-1].attrs['scale'] = (
            scaling.isel(scale=i, drop=True).isel(method=j, drop=True).item())
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0].objective
        else:
            last = owflop.history[-1].objective
            if last < best:
                best = last
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout,
                dim='xy'
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < owflop.rotor_diameter_adim / 10:
                break
        # calculate new layout
        scaling = scaling.isel(scale=i, drop=True) * scaler
        # swap in driving wind resources
        owflop._ds['direction_pmf'] = wind_resources.direction_pmf
        owflop._ds['wind_speed_cpmf'] = wind_resources.wind_speed_cpmf
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds.relative_deficit.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds.wake_loss_factor.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['unit_vector'] = (
            owflop._ds.unit_vector.isel(scale=i, drop=True)
                                  .isel(method=j, drop=True))
        owflop.calculate_relative_wake_loss_vector()
        down_step = (
            owflop.calculate_push_down_vector().mean(dim='wind_resource'))
        back_step = (
            owflop.calculate_push_back_vector().mean(dim='wind_resource'))
        cross_step = (
            owflop.calculate_push_cross_vector().mean(dim='wind_resource'))
        # throw steps in one big DataArray
        step = _xr.concat([down_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # generate the different step variants
        step = step * owflop.rotor_diameter_adim * multiplier * scaling
        # take the step
        _take_step(owflop, step)
        corrections = fix_constraints(owflop)
        iterations += 1
        # swap in objective wind resource again
        owflop._ds['direction_pmf'] = wind_resource.direction_pmf
        owflop._ds['wind_speed_cpmf'] = wind_resource.wind_speed_cpmf
