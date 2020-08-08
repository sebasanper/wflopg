import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS
from wflopg import create_site
from wflopg import create_constraint


def hexagonal(turbines, site_parcels, site_violation_distance, to_border,
              offset=None, angle=None):
    """Create hexagonal—so densest—packing to cover site

    The `to_border` function must be the one applicable for the site.

    The `offset` must be a list or array of two floats between 0 and one:
    the first is the fraction for the basic distance between columns;
    the second is the fraction for the basic distance between rows.

    """
    if offset is None:
        # create random offset
        offset = _np.random.random(2)
    if angle is None:
        # create random rotation matrix
        angle = _np.random.random() * _np.pi / 3  # hexgrid is π/3-symmetric
    cos_angle = _np.cos(angle)
    sin_angle = _np.sin(angle)
    rotation_matrix = _xr.DataArray(
        _np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]),
        coords=[('uv', ['u', 'v']), ('xy', COORDS['xy'])]
    )
    # initialize algorithm parameters
    max_turbines = 0
    factor = 1
    # Process parcels
    hex_parcels = create_site.parcels(
        site_parcels, -site_violation_distance, rotor_constraint_override=True)
    # create function that reports whether a turbine is inside the site
    hex_inside = create_constraint.inside_site(hex_parcels)
    #
    print("create hex layout: ", end='')
    #
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
        # apply offset
        offset_step = _xr.DataArray(
            offset * _np.array([x_step, y_step]), coords=[('uv', ['u', 'v'])])
        covering_layout += offset_step
        # apply rotation
        rotated_covering_layout = covering_layout.dot(rotation_matrix)
        # only keep turbines inside
        inside = hex_inside(rotated_covering_layout)
        dense_layout = rotated_covering_layout[inside['in_site']]
        max_turbines = len(dense_layout)
        print(max_turbines, end=' ')
        factor *= max_turbines / turbines
    #
    print()
    #
    dense_layout.attrs['hex_distance'] = x_step
    dense_layout.attrs['offset'] = offset
    dense_layout.attrs['rotation_angle'] = angle
    return dense_layout + to_border(dense_layout)


def fix_constraints(owflop, output=True):
    corrections = ''
    maybe_violations = True
    while maybe_violations:
        outside = ~owflop.inside(owflop._ds.layout)['in_site']
        any_outside = outside.any()
        if any_outside:
            if output:
                print('s', outside.values.sum(), sep='', end='')
            owflop.process_layout(
                owflop._ds.layout + owflop.to_border(owflop._ds.layout))
            corrections += 's'
        proximity_repulsion_step = (
            owflop.proximity_repulsion(
                owflop._ds.distance, owflop._ds.unit_vector)
        )
        too_close = proximity_repulsion_step is not None
        if too_close:
            if output:
                print('p', proximity_repulsion_step.attrs['violations'],
                      sep='', end='')
            owflop.process_layout(owflop._ds.layout + proximity_repulsion_step)
            corrections += 'p'
        if output:
            print(' ', end='')
        maybe_violations = too_close
    if output:
        print('\n', end='')
    return corrections
