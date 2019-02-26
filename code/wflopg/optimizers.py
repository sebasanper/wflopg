import numpy as np
import xarray as xr


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
        owflop.process_layout(owflop._ds['layout'] + step)
        owflop.calculate_geometry()
        # deal with any constraint violations in layout
        corrections = ''
        maybe_violations = True
        while maybe_violations:
            maybe_violations = False
            print('v', end=' ')
            is_outside = (
                not owflop.inside(owflop._ds['layout'])['in_site'].all())
            if is_outside:
                print('s', end=' ')
                outside = np.square(owflop._ds['layout']).sum(dim='xy') > 1
                owflop.process_layout(
                    owflop._ds['layout']
                    + owflop.to_border(owflop._ds['layout'])
                )
                owflop.calculate_geometry()
                corrections += 's'
            proximity_violated = (
                owflop.proximity_violation(owflop._ds['distance']))
            too_close = proximity_violated.any()
            if too_close:
                print('p', end=' ')
                owflop.process_layout(
                    owflop._ds['layout']
                    + owflop.proximity_repulsion(
                        proximity_violated,
                        owflop._ds['unit_vector'],
                        owflop._ds['distance']
                    )
                )
                owflop.calculate_geometry()
                corrections += 'p'
                maybe_violations = True
        iterations += 1


def pure_down(owflop, max_iterations=np.inf):
    """Optimize the layout using push-down only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    _iterate(owflop.calculate_push_down_vector, owflop, max_iterations)


def pure_back(owflop, max_iterations=np.inf):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    _iterate(owflop.calculate_push_back_vector, owflop, max_iterations)


def mixed_down_and_back(owflop, max_iterations=np.inf):
    """Optimize the layout using a mixture of push-down and push-back

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    def step_generator():
        return (owflop.calculate_push_down_vector()
                + owflop.calculate_push_back_vector()) / 2

    _iterate(step_generator, owflop, max_iterations)


def pure_cross(owflop, max_iterations=np.inf):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    _iterate(owflop.calculate_push_cross_vector, owflop, max_iterations)
