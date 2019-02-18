import numpy as np
import xarray as xr


def pure_push_down(owflop):
    """Optimize the layout for the given problem object using push-down only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    corrections = ''
    best = last = start = 0
    while last - best <= (start - best) / np.log2(len(owflop.history) + 2):
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
        step = owflop.calculate_push_down_vector()
        step -= step.mean(dim='target')  # remove any global shift
        owflop.process_layout(owflop._ds['layout'] + step)
        owflop.calculate_geometry()
        # deal with any constraint violations in layout
        corrections = ''
        some_outside, too_close_together = True, True
        while some_outside or too_close_together:
            some_outside = not owflop.inside(owflop._ds['layout']).all()
            if some_outside:
                outside = np.square(owflop._ds['layout']).sum(dim='xy') > 1
                step = owflop.to_border(owflop._ds['layout'])
                owflop.process_layout(
                    owflop._ds['layout']
                    + owflop.to_border(owflop._ds['layout'])
                )
                owflop.calculate_geometry()
                corrections += 's'
            proximity_violated = (
                owflop.proximity_violation(owflop._ds['distance']))
            too_close_together = proximity_violated.any()
            if too_close_together:
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
