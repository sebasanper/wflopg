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
    best = last = start = 0
    while ((iterations < max_iterations) and
           (last - best <= (start - best) / np.log2(len(owflop.history) + 2))):
        print(iterations, end=' ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        owflop.history.append(xr.Dataset())
        owflop.history[-1]['layout'] = owflop._ds['layout']
        owflop.history[-1]['objective'] = owflop.objective()
        owflop.history[-1].attrs['corrections'] = corrections
        if len(owflop.history) > 1:
            last = owflop.history[-1]['objective']
            if last < best:
                best = last
        else:
            best = last = start = owflop.history[0]['objective']
        # calculate new layout
        owflop.calculate_relative_wake_loss_vector()
        step = step_generator()
        step -= step.mean(dim='target')  # remove any global shift
        _take_step(owflop, step)
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            print('v', end=' ')
            is_outside = (
                not owflop.inside(owflop._ds['layout'])['in_site'].all())
            if is_outside:
                print('s', end=' ')
                _take_step(owflop, owflop.to_border(owflop._ds['layout']))
                corrections += 's'
            proximity_violated = (
                owflop.proximity_violation(owflop._ds['distance']))
            too_close = proximity_violated.any()
            if too_close:
                print('p', end=' ')
                _take_step(
                    owflop,
                    owflop.proximity_repulsion(
                        proximity_violated,
                        owflop._ds['unit_vector'],
                        owflop._ds['distance']
                    )
                )
                corrections += 'p'
            maybe_violations = is_outside & too_close
        iterations += 1


def _adaptive_iterate(step_generator, owflop, max_iterations):
    iterations = 0
    corrections = ''
    best = last = start = 0
    owflop._ds['layout'] = owflop._ds['context'] = (
        owflop._ds['layout']
        * xr.DataArray([1., 1., 1.], coords=[('scale', ['-', '0', '+'])]))
    owflop._ds['context'] = owflop._ds['context'].rename(target='source')
    owflop.calculate_geometry()
    scaler = xr.DataArray([.5, 1., 2.], coords=[('scale', ['-', '0', '+'])])
    scaling = xr.DataArray([1., 1., 1.], coords=[('scale', ['-', '0', '+'])])
    while ((iterations < max_iterations) and
           (last - best <= (start - best) / np.log2(len(owflop.history) + 2))):
        print(iterations, end=' ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        print(objectives)
        i = objectives.argmin()
        owflop.history.append(xr.Dataset())
        owflop.history[-1]['layout'] = owflop._ds['layout'].isel(scale=i)
        owflop.history[-1]['objective'] = objectives.isel(scale=i)
        owflop.history[-1].attrs['corrections'] = corrections
        if len(owflop.history) > 1:
            last = owflop.history[-1]['objective']
            if last < best:
                best = last
        else:
            best = last = start = owflop.history[0]['objective']
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
            print('v', end=' ')
            is_outside = (
                not owflop.inside(owflop._ds['layout'])['in_site'].all())
            if is_outside:
                print('s', end=' ')
                _take_step(owflop, owflop.to_border(owflop._ds['layout']))
                corrections += 's'
            proximity_violated = (
                owflop.proximity_violation(owflop._ds['distance']))
            too_close = proximity_violated.any()
            if too_close:
                print('p', end=' ')
                _take_step(
                    owflop,
                    owflop.proximity_repulsion(
                        proximity_violated,
                        owflop._ds['unit_vector'],
                        owflop._ds['distance']
                    )
                )
                corrections += 'p'
            maybe_violations = is_outside & too_close
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
