import sys as _sys
import numpy as _np
import xarray as _xr
import matplotlib.pyplot as _plt
import matplotlib.gridspec as _gs

import wflopg.visualization as vis
from wflopg.create_layout import fix_constraints
from wflopg.helpers import rss


def _update_history(owflop, arrays={}, attrs={}):
    owflop.history.append(_xr.Dataset())
    for name, array in arrays.items():
        owflop.history[-1][name] = array
    for name, attr in attrs.items():
        owflop.history[-1].attrs[name] = attr

def _setup_visualization(owflop):
    axes = {}
    fig = _plt.figure()
    grid = _gs.GridSpec(3, 5)
    axes['windrose'] = fig.add_subplot(grid[0, :2], polar=True)
    vis.draw_windrose(axes['windrose'], owflop._ds.direction_pmf)
    axes['convergence'] = fig.add_subplot(grid[1, :2])
    axes['step_size'] = fig.add_subplot(grid[2, :2],
                                        sharex=axes['convergence'])
    axes['layout'] = fig.add_subplot(grid[:, 2:])
    vis.site_setup(axes['layout'])
    vis.draw_turbines(axes['layout'], owflop, owflop.history[0].layout,
                        proximity=True, in_or_out=True)
    vis.draw_boundaries(axes['layout'], owflop)
    grid.tight_layout(fig)
    _plt.pause(.10)
    return axes

def _iterate_visualization(axes, owflop):
    axes['convergence'].clear()
    vis.draw_convergence(axes['convergence'], owflop.history)
    axes['step_size'].clear()
    vis.draw_step_size(axes['step_size'], owflop.history)
    axes['layout'].clear()
    vis.site_setup(axes['layout'])
    vis.connect_layouts(axes['layout'], [ds.layout for ds in owflop.history])
    vis.draw_turbines(axes['layout'], owflop, owflop.history[0].layout)
    vis.draw_turbines(axes['layout'], owflop, owflop.history[-1].layout,
                      proximity=True, in_or_out=True)
    vis.draw_boundaries(axes['layout'], owflop)
    _plt.pause(0.1)

def _step_generator(owflop, method):
    if method == 'away':
        return owflop.calculate_push_away_vector()
    elif method == 'back':
        return owflop.calculate_push_back_vector()
    elif method == 'cross':
        return owflop.calculate_push_cross_vector()
    elif method == 'away_and_back_mix':
        return (owflop.calculate_push_away_vector()
                + owflop.calculate_push_back_vector()) / 2
    else:
        raise ValueError(f"Method ‘{method}’ is unknown.")


def step_iterator(owflop, methods=None, max_iterations=_sys.maxsize,
                  multiplier=1, scaling=True, wake_spreading=False,
                  visualize=False):

    if methods is None:
        methods = ['away', 'back', 'cross']
    elif isinstance(methods, str):
        methods = [methods]

    if scaling is True:
        scaling = [.8, 1.1]
    elif scaling is False:
        scaling = [1]
    scaler = (
        _xr.DataArray(_np.float64(scaling), dims=('scale',))
        * _xr.DataArray(_np.ones(len(methods)), coords=[('method', methods)])
    )

    initial_multiplier = multiplier
    if wake_spreading:
        spread_multiplier = initial_multiplier
    else:
        spread_multiplier = 0
    def spread_factor(spread_multiplier):
        return (1 + 3 * (spread_multiplier - 0)
                      / (initial_multiplier - 0))

    owflop.calculate_deficit(spread_factor(spread_multiplier))
    owflop.calculate_power()
    _update_history(owflop,
                    arrays={'layout': owflop._ds.layout,
                            'objective': owflop.objective(),
                            'objective_bound': _np.nan},
                    attrs={'corrections': '',
                           'max_step': _np.nan,
                           'actual_step': _np.nan,
                           'method': '',
                           'spread': _np.nan})
    best = start = owflop.history[0].objective
    if visualize:
        axes = _setup_visualization(owflop)
    for iteration in range(1, max_iterations+1):
        print(iteration, end=': ')
        owflop.process_layout(owflop.history[-1].layout)
        spread = spread_factor(spread_multiplier)
        owflop.calculate_deficit(spread)
        # calculate step
        owflop.calculate_relative_wake_loss_vector()
        step = _xr.concat(
            [_step_generator(owflop, method) for method in methods], 'method')
        # remove any global shift
        step -= step.mean(dim='target')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        del distance
        # take step
        multiplier = scaler * multiplier
        step = step * multiplier * owflop.rotor_diameter_adim
        owflop.process_layout(owflop._ds.layout + step)
        del step
        # fix any constraint violation
        corrections = fix_constraints(owflop)
        # evaluate new layout
        owflop._ds = owflop._ds.chunk({'scale': 1, 'method': 1})
        owflop.calculate_deficit(spread)
        owflop.calculate_power()
        owflop._ds.load()
        i = owflop.objective().argmin('scale')
        owflop._ds = owflop._ds.isel(scale=i, drop=True)
        j = owflop.objective().argmin('method').item()
        owflop._ds = owflop._ds.isel(method=j, drop=True)
        layout = owflop._ds.layout
        current = owflop.objective()
        print(f"(wfl: {_np.round(current.item() * 100, 4)};",
              f"spread: {_np.round(spread, 2)})", sep=' ')
        bound = best + (start - best) / iteration
        multiplier = _np.maximum(multiplier.isel(scale=i, drop=True),
                                 minimal_multiplier)
        current_multiplier = multiplier.isel(method=j, drop=True)
        max_distance = (rss(layout - owflop.history[-1].layout, dim='xy').max()
                        / owflop.rotor_diameter_adim)
        _update_history(owflop,
                        arrays={'layout': layout,
                                'objective': current,
                                'objective_bound': bound},
                        attrs={'corrections': corrections,
                               'max_step': current_multiplier,
                               'actual_step': max_distance,
                               'method': methods[j],
                               'spread': spread_multiplier})
        if visualize:
            _iterate_visualization(axes, owflop)
        # check best layout and criteria for early termination
        if current < best:
            best = current
        elif current > bound:
            break
        if wake_spreading:
            if current_multiplier < spread_multiplier:
                weight = _np.log2(iteration + 1)
                spread_multiplier = (
                    (spread_multiplier * weight + current_multiplier)
                    / (weight + 1)
                )
            else:
                # force reduction of spread multiplier to avoid getting stuck
                spread_multiplier /= 10**0.01
