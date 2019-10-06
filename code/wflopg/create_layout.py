import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS
from wflopg import create_site
from wflopg import create_constraint


def hexagonal(turbines, site_parcels, site_violation_distance, to_border):
    """Create hexagonal—so densest—packing to cover site
    
    The `to_border` function must be the one applicable for the site.
    
    """
    max_turbines = 0
    factor = 1
    # Process parcels
    hex_parcels = create_site.parcels(
        site_parcels, -site_violation_distance, rotor_constraint_override=True)
    # create function that reports whether a turbine is inside the site
    hex_inside = create_constraint.inside_site(hex_parcels)
    while max_turbines != turbines:
        x_step = _np.sqrt(factor / turbines) * 2
        y_step = x_step * _np.sqrt(3) / 2
        n = _np.ceil(1 / x_step)
        m = _np.ceil(1 / y_step)
        xs = _np.arange(-n, n+1) * x_step
        ys = _np.arange(-m, m+1) * y_step
        mg = _np.meshgrid(xs, ys)
        mg[0] = (mg[0].T + (_np.arange(-m, m+1) % 2) * x_step / 2).T
        covering_layout = _xr.DataArray(
            _np.stack([mg[0].ravel(), mg[1].ravel()], axis=-1),
            dims=['target', 'uv'], coords={'uv': ['u', 'v']}
        )
        # add random offset
        offset = _xr.DataArray(
            _np.random.random(2) * _np.array([x_step, y_step]),
            coords=[('uv', ['u', 'v'])]
        )
        covering_layout += offset
        # rotate over random angle
        angle = _np.random.random() * _np.pi / 3 # hexgrid is π/3-symmetric
        cos_angle = _np.cos(angle)
        sin_angle = _np.sin(angle)
        rotation_matrix = _xr.DataArray(
            _np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]),
            coords=[('uv', ['u', 'v']), ('xy', COORDS['xy'])]
        )
        rotated_covering_layout = covering_layout.dot(rotation_matrix)
        # only keep turbines inside
        inside = hex_inside(rotated_covering_layout)
        dense_layout = rotated_covering_layout[inside['in_site']]
        dense_layout.attrs['hex_distance'] = x_step
        max_turbines = len(dense_layout)
        factor *= max_turbines / turbines
    return dense_layout + to_border(dense_layout)
