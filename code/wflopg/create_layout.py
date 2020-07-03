import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS
import wflopg.helpers as _hs
from wflopg import create_site
from wflopg import create_constraint


class Layout():
    """A wind farm layout object."""
    def __init__(self, filename=None, **kwargs):
        """Create a wind farm layout object.
        
        Parameters
        ----------
        filename
            The path to a file containing a wind farm layout
            description satisfying the wind farm layout schema.
        kwargs
            A (nested) `dict` containing a (partial) description
            of a wind farm layout according to the wind farm layout
            schema. It can be used both to override elements of
            the description given in `filename` or to define
            a wind farm layout in its entirety.
        
        """
        # TODO: add reference to actual wind farm layout schema
        layout_dict = {}
        if filename is not None:
            with open(filename) as f:
                layout_dict.update(_hs.yaml_load(f))
        layout_dict.update(kwargs)
        # TODO: check layout_dict with schema using jsonschema
        
        layout_array = _np.array(layout_dict['layout'])
        self.layout = _xr.Dataset(
            coords={'loc': range(len(layout_array))})
        for name, index in {('x', 0), ('y', 1)}:
            self.layout[name] = ('loc', layout_array[:, index])
        for name in {'target', 'source', 'movable'}:
            self.layout[name] = _xr.full_like(self.layout.loc, True)
            
        # TODO: add other layout_dict properties as attributes to self?

    def initialize_relative_positions(self, frm, to):
        """Add and initialize relative positions to wind farm layout object.
        
        Parameters
        ----------
        frm
            Boolean array identifying locations
            to calculate relative positions from.
        to
            Boolean array identifying locations
            to calculate relative positions to.
        
        """
        self.rel = _xr.Dataset(coords={'frm': self.layout.loc[frm],
                                       'to': self.layout.loc[to]})
        for z in {'x', 'y'}:
            da = self.layout[z]
            self.rel[z] = (
                da.sel(at=self.rel['to']).rename({'loc': 'to'})
                - da.sel(at=self.rel['frm']).rename({'loc': 'frm'})
            )
    
    def _has_rel_check(self):
        if not hasattr(self, 'rel'):
            raise AttributeError(
                "Call the ‘initialize_relative_positions’ method first.")

    def get_distances(self):
        """Get distances between locations."""
        self._has_rel_check()
        if 'distance' not in self.rel:
            self.rel['distance'] = (
                _np.sqrt(_np.square(self.rel.x) + _np.square(self.rel.y)))
        return self.rel.distance
        
    def get_angles(self):
        """Get angles between locations."""
        self._has_rel_check()
        if 'angle' not in self.rel:
            self.rel['angle'] = _np.atan2(self.rel.y, self.rel.x)
        return self.rel.angle
        

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
        angle = _np.random.random() * _np.pi / 3  # hexgrid is π/3-symmetric
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
