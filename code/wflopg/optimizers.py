import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import wflopg.visualization as vis


def _take_step(owflop, step):
    owflop._ds['layout'] = owflop._ds['context'] = (
        owflop._ds['layout'] + step)
    owflop._ds['context'] = owflop._ds.context.rename(target='source')
    owflop.calculate_geometry()


def _iterate(step_generator, owflop, max_iterations, step_normalizer):
    site_rotor_diameter = (owflop.rotor_radius / owflop.site_radius) * 2
    iterations = 0
    corrections = ''
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (last - best
                > (start - best) / np.log2(len(owflop.history) + 2)):
                break
        print('(', iterations, sep='', end=':')
        owflop.calculate_deficit()
        owflop.calculate_power()
        owflop.history.append(xr.Dataset())
        owflop.history[-1]['layout'] = owflop._ds.layout
        owflop.history[-1]['objective'] = owflop.objective()
        owflop.history[-1].attrs['corrections'] = corrections
        if len(owflop.history) == 1:  # first run
            best = last = start = owflop.history[0]['objective']
        else:
            last = owflop.history[-1]['objective']
            if last < best:
                best = last
            distance_from_previous = np.sqrt(
                np.square(
                    owflop.history[-1]['layout'] - owflop.history[-2]['layout']
                ).sum(dim='xy')
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < site_rotor_diameter / 10:
                break
        # calculate new layout
        owflop.calculate_relative_wake_loss_vector()
        step = step_generator()
        # normalize the step to the largest pseudo-gradient
        distance = np.sqrt(np.square(step).sum(dim='xy'))
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        step *= step_normalizer
        _take_step(owflop, step)
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            outside = ~owflop.inside(owflop._ds.layout)['in_site']
            any_outside = outside.any()
            if any_outside:
                print('s', outside.values.sum(), sep='', end='')
                _take_step(owflop, owflop.to_border(owflop._ds.layout))
                corrections += 's'
            proximity_repulsion_step = (
                owflop.proximity_repulsion(
                    owflop._ds.distance, owflop._ds.unit_vector)
            )
            too_close = proximity_repulsion_step is not None
            if too_close:
                print('p', proximity_repulsion_step.attrs['violations'],
                      sep='', end='')
                _take_step(owflop, proximity_repulsion_step)
                corrections += 'p'
            print(',', end='')
            maybe_violations = too_close
        print(')', end=' ')
        iterations += 1


def _adaptive_iterate(step_generator, owflop, max_iterations, step_normalizer,
                      scaler=[.5, 1.1], visualize=False):
    site_rotor_diameter = (owflop.rotor_radius / owflop.site_radius) * 2
    if visualize:
        fig = plt.figure()
        grid = gs.GridSpec(3, 5)
        ax_windrose = fig.add_subplot(grid[0, :2], polar=True)
        ax_windrose.set_aspect(1.0)
        ax_windrose.set_theta_zero_location("N")
        ax_windrose.set_theta_direction(-1)
        ax_windrose.set_ylim(
                0, 1.1 * owflop._ds.direction_pmf.max().values.item())
        ax_windrose.bar(owflop._ds.direction / 360 * 2 * np.pi,
                        owflop._ds.direction_pmf,
                        color='b', width=2 * np.pi / len(owflop._ds.direction))
        ax_convergence = fig.add_subplot(grid[1, :2])
        ax_convergence.set_xlim(-1, max_iterations + 1)
        ax_convergence.set_ylim(0, 100)
        ax_scaling = fig.add_subplot(grid[2, :2], sharex=ax_convergence)
        ax_layout = fig.add_subplot(grid[:, 2:])
        ax_layout.set_aspect('equal')
        ax_layout.set_axis_off()
        ax_layout.set_xlim(-1.01, 1.01)
        ax_layout.set_ylim(-1.01, 1.01)
        vis.draw_turbines(ax_layout, owflop, owflop._ds.layout,
                          proximity=True, in_or_out=True)
        vis.draw_boundaries(ax_layout, owflop)
        grid.tight_layout(fig)
        plt.pause(.10)
    scale_coord = ('scale', ['-', '+'])
    iterations = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * xr.DataArray([1, 1], coords=[scale_coord])
    )
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    scaler = xr.DataArray(scaler, coords=[scale_coord])
    scaling = xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (last - best
                > (start - best) / np.log2(len(owflop.history) + 2)):
                break
        print('(', iterations, sep='', end=':')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin()
        owflop.history.append(xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True))
        owflop.history[-1]['objective'] = objectives.isel(scale=i, drop=True)
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['scale'] = scaling[i].values.item()
        if visualize:
            ax_convergence.clear()
            ax_convergence.plot(
                    100 * np.array([ds.objective for ds in owflop.history]))
            ax_scaling.clear()
            ax_scaling.semilogy(
                    np.array([ds.scale for ds in owflop.history]), '.')
            ax_layout.clear()
            ax_layout.set_aspect('equal')
            ax_layout.set_axis_off()
            ax_layout.set_xlim(-1.01, 1.01)
            ax_layout.set_ylim(-1.01, 1.01)
            vis.connect_layouts(ax_layout,
                                [ds.layout for ds in owflop.history])
            vis.draw_turbines(ax_layout, owflop, owflop.history[0].layout)
            vis.draw_turbines(ax_layout, owflop, owflop.history[-1].layout,
                              proximity=True, in_or_out=True)
            vis.draw_boundaries(ax_layout, owflop)
            plt.pause(0.1)
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0]['objective']
        else:
            last = owflop.history[-1]['objective']
            if last < best:
                best = last
            distance_from_previous = np.sqrt(
                np.square(
                    owflop.history[-1]['layout'] - owflop.history[-2]['layout']
                ).sum(dim='xy')
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < site_rotor_diameter / 10:
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
        distance = np.sqrt(np.square(step).sum(dim='xy'))
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        step *= step_normalizer
        step = step * scaling  # generate the different step variants
        _take_step(owflop, step)
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            outside = ~owflop.inside(owflop._ds.layout)['in_site']
            any_outside = outside.any()
            if any_outside:
                print('s', outside.values.sum(), sep='', end='')
                _take_step(owflop, owflop.to_border(owflop._ds.layout))
                corrections += 's'
            proximity_repulsion_step = (
                owflop.proximity_repulsion(
                    owflop._ds.distance, owflop._ds.unit_vector)
            )
            too_close = proximity_repulsion_step is not None
            if too_close:
                print('p', proximity_repulsion_step.attrs['violations'],
                      sep='', end='')
                _take_step(owflop, proximity_repulsion_step)
                corrections += 'p'
            print(',', end='')
            maybe_violations = too_close
        print(')', end=' ')
        iterations += 1


def pure_down(owflop, max_iterations=np.inf,
              scaling=False, scaler=[.5, 1.1], multiplier=3, visualize=False):
    """Optimize the layout using push-down only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    step_normalizer = (owflop.rotor_radius / owflop.site_radius) * 2
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_down_vector, owflop, max_iterations,
            step_normalizer * multiplier, scaler=scaler, visualize=visualize
        )
    else:
        _iterate(
            owflop.calculate_push_down_vector, owflop,
            max_iterations, step_normalizer * multiplier
        )


def pure_back(owflop, max_iterations=np.inf,
              scaling=False, scaler=[.5, 1.1], multiplier=3, visualize=False):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    step_normalizer = (owflop.rotor_radius / owflop.site_radius) * 2
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_back_vector, owflop, max_iterations,
            step_normalizer * multiplier, scaler=scaler, visualize=visualize
        )
    else:
        _iterate(
            owflop.calculate_push_back_vector, owflop,
            max_iterations, step_normalizer * multiplier
        )


def mixed_down_and_back(owflop, max_iterations=np.inf,
                        scaling=False, scaler=[.5, 1.1], multiplier=3,
                        visualize=False):
    """Optimize the layout using a mixture of push-down and push-back

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    step_normalizer = (owflop.rotor_radius / owflop.site_radius) * 2

    def step_generator():
        return (owflop.calculate_push_down_vector()
                + owflop.calculate_push_back_vector()) / 2

    if scaling:
        _adaptive_iterate(
            step_generator, owflop, max_iterations,
            step_normalizer * multiplier, scaler=scaler, visualize=visualize)
    else:
        _iterate(step_generator, owflop, max_iterations,
                 step_normalizer * multiplier)


def pure_cross(owflop, max_iterations=np.inf, scaling=False,
               scaler=[.5, 1.1], multiplier=3, visualize=False):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    step_normalizer = (owflop.rotor_radius / owflop.site_radius) * 2
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_cross_vector, owflop, max_iterations,
            step_normalizer * multiplier, scaler=scaler,
            visualize=visualize
        )
    else:
        _iterate(
            owflop.calculate_push_cross_vector, owflop,
            max_iterations, step_normalizer * multiplier
        )


def multi_adaptive(owflop, max_iterations=np.inf,
                   scaler=[.5, 1.1], multiplier=3,
                   only_above_average=False, visualize=False):
    if visualize:
        fig = plt.figure()
        grid = gs.GridSpec(3, 5)
        ax_windrose = fig.add_subplot(grid[0, :2], polar=True)
        ax_windrose.set_aspect(1.0)
        ax_windrose.set_theta_zero_location("N")
        ax_windrose.set_theta_direction(-1)
        ax_windrose.set_ylim(
                0, 1.1 * owflop._ds.direction_pmf.max().values.item())
        ax_windrose.bar(owflop._ds.direction / 360 * 2 * np.pi,
                        owflop._ds.direction_pmf,
                        color='b', width=2 * np.pi / len(owflop._ds.direction))
        ax_convergence = fig.add_subplot(grid[1, :2])
        ax_convergence.set_xlim(-1, max_iterations + 1)
        ax_convergence.set_ylim(0, 100)
        ax_scaling = fig.add_subplot(grid[2, :2], sharex=ax_convergence)
        ax_layout = fig.add_subplot(grid[:, 2:])
        ax_layout.set_aspect('equal')
        ax_layout.set_axis_off()
        ax_layout.set_xlim(-1.01, 1.01)
        ax_layout.set_ylim(-1.01, 1.01)
        vis.draw_turbines(ax_layout, owflop, owflop._ds.layout,
                          proximity=True, in_or_out=True)
        vis.draw_boundaries(ax_layout, owflop)
        grid.tight_layout(fig)
        plt.pause(.10)
    site_rotor_diameter = (owflop.rotor_radius / owflop.site_radius) * 2
    scale_coord = ('scale', ['-', '+'])
    method_coord = ('method', ['down', 'back', 'cross'])
    iterations = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * xr.DataArray(
            [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
    )
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    scaler = xr.DataArray(scaler, coords=[scale_coord])
    scaling = xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (last - best
                > (start - best) / np.log2(len(owflop.history) + 2)):
                break
        print('(', iterations, sep='', end=':')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin(dim='scale')
        j = objectives.min(dim='scale').argmin('method').values.item()
        owflop.history.append(xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
        )
        owflop.history[-1]['objective'] = (
            objectives.isel(scale=i, drop=True).isel(method=j, drop=True))
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        owflop.history[-1].attrs['scale'] = (
            scaling.isel(scale=i, drop=True)
                   .isel(method=j, drop=True).values.item()
        )
        if visualize:
            ax_convergence.clear()
            ax_convergence.plot(
                    100 * np.array([ds.objective for ds in owflop.history]))
            ax_scaling.clear()
            ax_scaling.semilogy(
                    np.array([ds.scale for ds in owflop.history]), '.')
            ax_layout.clear()
            ax_layout.set_aspect('equal')
            ax_layout.set_axis_off()
            ax_layout.set_xlim(-1.01, 1.01)
            ax_layout.set_ylim(-1.01, 1.01)
            vis.connect_layouts(ax_layout,
                                [ds.layout for ds in owflop.history])
            vis.draw_turbines(ax_layout, owflop, owflop.history[0].layout)
            vis.draw_turbines(ax_layout, owflop, owflop.history[-1].layout,
                              proximity=True, in_or_out=True)
            vis.draw_boundaries(ax_layout, owflop)
            plt.pause(0.1)
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0]['objective']
        else:
            last = owflop.history[-1]['objective']
            if last < best:
                best = last
            distance_from_previous = np.sqrt(
                np.square(
                    owflop.history[-1]['layout'] - owflop.history[-2]['layout']
                ).sum(dim='xy')
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < site_rotor_diameter / 10:
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
        step = xr.concat([down_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = np.sqrt(np.square(step).sum(dim='xy'))
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # only take above average steps
        if only_above_average:
            mean_distance = distance.mean(dim='target')
            step *= (distance > mean_distance)
        # generate the different step variants
        step = step * site_rotor_diameter * multiplier * scaling
        # take the step
        _take_step(owflop, step)
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            outside = ~owflop.inside(owflop._ds.layout)['in_site']
            any_outside = outside.any()
            if any_outside:
                print('s', outside.values.sum(), sep='', end='')
                _take_step(owflop, owflop.to_border(owflop._ds.layout))
                corrections += 's'
            proximity_repulsion_step = (
                owflop.proximity_repulsion(
                    owflop._ds.distance, owflop._ds.unit_vector)
            )
            too_close = proximity_repulsion_step is not None
            if too_close:
                print('p', proximity_repulsion_step.attrs['violations'],
                      sep='', end='')
                _take_step(owflop, proximity_repulsion_step)
                corrections += 'p'
            print(',', end='')
            maybe_violations = too_close
        print(')', end=' ')
        iterations += 1


def method_chooser(owflop, max_iterations=np.inf):
    site_rotor_diameter = (owflop.rotor_radius / owflop.site_radius) * 2
    method_coord = ('method', ['down', 'back', 'cross'])
    iterations = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * xr.DataArray([1, 1, 1], coords=[method_coord]))
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (last - best
                > (start - best) / np.log2(len(owflop.history) + 2)):
                break
        print('(', iterations, sep='', end=':')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        j = objectives.argmin('method').values.item()
        owflop.history.append(xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(method=j, drop=True))
        owflop.history[-1]['objective'] = objectives.isel(method=j, drop=True)
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0]['objective']
        else:
            last = owflop.history[-1]['objective']
            if last < best:
                best = last
            distance_from_previous = np.sqrt(
                np.square(
                    owflop.history[-1]['layout'] - owflop.history[-2]['layout']
                ).sum(dim='xy')
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < site_rotor_diameter / 10:
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
        step = xr.concat([down_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = np.sqrt(np.square(step).sum(dim='xy'))
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # take the step, one rotor diameter for the largest pseudo-gradient
        _take_step(owflop, step * site_rotor_diameter)
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            outside = ~owflop.inside(owflop._ds.layout)['in_site']
            any_outside = outside.any()
            if any_outside:
                print('s', outside.values.sum(), sep='', end='')
                _take_step(owflop, owflop.to_border(owflop._ds.layout))
                corrections += 's'
            proximity_repulsion_step = (
                owflop.proximity_repulsion(
                    owflop._ds.distance, owflop._ds.unit_vector)
            )
            too_close = proximity_repulsion_step is not None
            if too_close:
                print('p', proximity_repulsion_step.attrs['violations'],
                      sep='', end='')
                _take_step(owflop, proximity_repulsion_step)
                corrections += 'p'
            print(',', end='')
            maybe_violations = too_close
        print(')', end=' ')
        iterations += 1
