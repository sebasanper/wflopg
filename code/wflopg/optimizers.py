import sys as _sys
import numpy as _np
import xarray as _xr
import matplotlib.pyplot as _plt
import matplotlib.gridspec as _gs

import wflopg.visualization as vis
from wflopg.create_layout import _take_step, fix_constraints
from wflopg.helpers import rss


def _update_history(owflop, layout,
                    objective, bound=_np.nan, corrections='',
                    multiplier=1, method=''):
    owflop.history.append(_xr.Dataset())
    owflop.history[-1]['layout'] = layout
    owflop.history[-1]['objective'] = objective
    owflop.history[-1]['objective_bound'] = bound
    owflop.history[-1].attrs['corrections'] = corrections
    owflop.history[-1].attrs['scale'] = multiplier
    owflop.history[-1].attrs['method'] = method

def _setup_visualization(owflop):
    axes = {}
    fig = _plt.figure()
    grid = _gs.GridSpec(3, 5)
    axes['windrose'] = fig.add_subplot(grid[0, :2], polar=True)
    vis.draw_windrose(axes['windrose'], owflop._ds.direction_pmf)
    axes['convergence'] = fig.add_subplot(grid[1, :2])
    axes['scaling'] = fig.add_subplot(grid[2, :2], sharex=axes['convergence'])
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
    axes['scaling'].clear()
    vis.draw_scaling(axes['scaling'], owflop.history)
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
                  multiplier=1, scaling=True,
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
    
    owflop.calculate_geometry()
    owflop.calculate_deficit()
    owflop.calculate_power()
    _update_history(owflop, owflop._ds.layout, owflop.objective())
    best = start = owflop.history[0].objective
    if visualize:
        axes = _setup_visualization(owflop)
    for iteration in range(1, max_iterations+1):
        print(iteration, end=': ')
        owflop._ds['layout'] = owflop.history[-1].layout
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        owflop.calculate_geometry()
        owflop.calculate_deficit()
        # calculate step
        owflop.calculate_relative_wake_loss_vector()
        step = _xr.concat(
            [_step_generator(owflop, method) for method in methods], 'method')
        # remove any global shift
        step -= step.mean(dim='target')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # take step
        multiplier = scaler * multiplier
        step = step * multiplier * owflop.rotor_diameter_adim
        _take_step(owflop, step)
        # fix any constraint violation
        corrections = fix_constraints(owflop)
        # evaluate new layout
        owflop.calculate_deficit()
        owflop.calculate_power()
        i = owflop.objective().argmin('scale')
        owflop._ds = owflop._ds.isel(scale=i, drop=True)
        j = owflop.objective().argmin('method').item()
        owflop._ds = owflop._ds.isel(method=j, drop=True)
        layout = owflop._ds.layout
        current = owflop.objective()
        bound = best + (start - best) / iteration
        multiplier = multiplier.isel(scale=i)
        _update_history(owflop, layout,
                        current, bound, corrections,
                        multiplier.isel(method=j), methods[j])
        if visualize:
            _iterate_visualization(axes, owflop)
        # check best layout and criteria for early termination
        if current < best:
            best = current
        else:
            if current > bound:
                break
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout, dim='xy'
            ) / owflop.rotor_diameter_adim
            if distance_from_previous.max() < .1:
                break
