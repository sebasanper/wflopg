import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS


def layout(rotor_radius, num_dists, num_dirs):
    """Create layout of points where the pre-averaged model will be calculated
    
    The rotor radius must be the site-adimensional rotor radius.
    
    """
    r = _np.linspace(2 * rotor_radius, 2, num_dists)
    r = _xr.DataArray(r, coords=[('r', r)])
    θ = 2 * _np.pi * _np.linspace(0, 1, num_dirs, endpoint=False)
    θ = _xr.DataArray(θ, coords=[('θ', θ)])
    x = (r * _np.cos(θ)).stack(target=('r', 'θ'))
    y = (r * _np.sin(θ)).stack(target=('r', 'θ'))
    return _xr.concat(
        [x, y], _xr.DataArray(COORDS['xy'], dims=['xy'], name='xy')
    ).transpose()
    
    
def context():
    return _xr.DataArray([[0, 0]],
                         dims=['source', 'xy'], coords={'xy': COORDS['xy']})
