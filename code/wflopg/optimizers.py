import numpy as np
import xarray as xr


def _take_step(owflop, step):
    owflop._ds['layout'] = owflop._ds['context'] = (
        owflop._ds['layout'] + step)
    owflop._ds['context'] = owflop._ds['context'].rename(target='source')
    owflop.calculate_geometry()


def _iterate(step_generator, owflop, max_iterations):
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
        owflop.history[-1]['layout'] = owflop._ds['layout']
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
            # stop iterating if the largest step is smaller than 1 m
            if distance_from_previous.max() * owflop.site_radius < 1:
                break
        # calculate new layout
        owflop.calculate_relative_wake_loss_vector()
        step = step_generator()
        step -= step.mean(dim='target')  # remove any global shift
        _take_step(owflop, step)
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            outside = ~owflop.inside(owflop._ds['layout'])['in_site']
            any_outside = outside.any()
            if any_outside:
                print('s', outside.values.sum(), sep='', end='')
                _take_step(owflop, owflop.to_border(owflop._ds['layout']))
                corrections += 's'
            proximity_violated = (
                owflop.proximity_violation(owflop._ds['distance']))
            too_close = proximity_violated.any()
            if too_close:
                print('p', proximity_violated.values.sum(), sep='', end='')
                _take_step(
                    owflop,
                    owflop.proximity_repulsion(
                        proximity_violated,
                        owflop._ds['unit_vector'],
                        owflop._ds['distance']
                    )
                )
                corrections += 'p'
            print(',', end='')
            maybe_violations = any_outside & too_close
        print(')', end=' ')
        iterations += 1


def _adaptive_iterate(step_generator, owflop, max_iterations):
    scale_coord = ('scale', ['-', '+'])
    iterations = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds['layout'] * xr.DataArray([1, 1], coords=[scale_coord])
    )
    owflop._ds['context'] = owflop._ds['layout'].rename(target='source')
    owflop.calculate_geometry()
    scaler = xr.DataArray([2/3, 6/5], coords=[scale_coord])
    scaling = xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        print('(', iterations, sep='', end=':')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin()
        owflop.history.append(xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds['layout'].isel(scale=i, drop=True))
        owflop.history[-1]['objective'] = objectives.isel(scale=i, drop=True)
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['scale'] = scaling[i].values.item()
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
            # stop iterating if the largest step is smaller than 1 m
            if distance_from_previous.max() * owflop.site_radius < 1:
                break
        # calculate new layout
        scaling = scaling.isel(scale=i) * scaler
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds['relative_deficit'].isel(scale=i))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds['wake_loss_factor'].isel(scale=i))
        owflop._ds['unit_vector'] = owflop._ds['unit_vector'].isel(scale=i)
        owflop.calculate_relative_wake_loss_vector()
        step = step_generator()
        step -= step.mean(dim='target')  # remove any global shift
        step = step * scaling  # generate the different step variants
        _take_step(owflop, step)
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            outside = ~owflop.inside(owflop._ds['layout'])['in_site']
            any_outside = outside.any()
            if any_outside:
                print('s', outside.values.sum(), sep='', end='')
                _take_step(owflop, owflop.to_border(owflop._ds['layout']))
                corrections += 's'
            proximity_violated = (
                owflop.proximity_violation(owflop._ds['distance']))
            too_close = proximity_violated.any()
            if too_close:
                print('p', proximity_violated.values.sum(), sep='', end='')
                _take_step(
                    owflop,
                    owflop.proximity_repulsion(
                        proximity_violated,
                        owflop._ds['unit_vector'],
                        owflop._ds['distance']
                    )
                )
                corrections += 'p'
            print(',', end='')
            maybe_violations = any_outside & too_close
        print(')', end=' ')
        iterations += 1


def pure_down(owflop, max_iterations=np.inf, scaling=False):
    """Optimize the layout using push-down only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_down_vector, owflop, max_iterations)
    else:
        _iterate(owflop.calculate_push_down_vector, owflop, max_iterations)


def pure_back(owflop, max_iterations=np.inf, scaling=False):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_back_vector, owflop, max_iterations)
    else:
        _iterate(owflop.calculate_push_back_vector, owflop, max_iterations)


def mixed_down_and_back(owflop, max_iterations=np.inf, scaling=False):
    """Optimize the layout using a mixture of push-down and push-back

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    def step_generator():
        return (owflop.calculate_push_down_vector()
                + owflop.calculate_push_back_vector()) / 2

    if scaling:
        _adaptive_iterate(step_generator, owflop, max_iterations)
    else:
        _iterate(step_generator, owflop, max_iterations)


def pure_cross(owflop, max_iterations=np.inf, scaling=False):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    if scaling:
        _adaptive_iterate(
            owflop.calculate_push_cross_vector, owflop, max_iterations)
    else:
        _iterate(owflop.calculate_push_cross_vector, owflop, max_iterations)
