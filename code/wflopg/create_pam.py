import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS


def layout(minimal_proximity, num_dists, num_dirs):
    """Create layout of points where the pre-averaged model will be calculated

    minimal_proximity must be in site-adimensional units.

    """
    r = _np.linspace(minimal_proximity, 2, num_dists)
    r = _xr.DataArray(r, coords=[('r', r)])
    θ = 2 * _np.pi * _np.linspace(0, 1, num_dirs, endpoint=False)
    θ = _xr.DataArray(θ, coords=[('θ', θ)])
    x = (r * _np.cos(θ)).stack(target=('r', 'θ'))
    y = (r * _np.sin(θ)).stack(target=('r', 'θ'))
    return _xr.concat(
        [x, y], _xr.DataArray(COORDS['xy'], dims=['xy'], name='xy')
    ).transpose()


def context():
    """Create the context to calculate the pre-averaged model

    This returns a context with a single turbine at the origin.

    """
    return _xr.DataArray([[0., 0.]],
                         dims=['source', 'xy'], coords={'xy': COORDS['xy']})


def compute(owflop):
    """Compute the pre-averaged model

    It is assumed that the provided problem object has been created with the
    pam layout type specified.

    """
    owflop.calculate_wakeless_power()
    owflop.calculate_geometry()
    owflop.calculate_deficit()
    owflop.calculate_power()
    return owflop._ds.expected_wake_loss_factor.unstack('target')


def deficits(pam, positions):
    """Compute the reciprocal deficits for a given sets of positions"""
    raise NotImplementedError()
