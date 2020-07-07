import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS
import wflopg.helpers as _hs
from wflopg import create_site
from wflopg import create_constraint


class Layout():
    """A wind farm layout object.
        
    Parameters
    ----------
    filename : str
        The path to a file containing a wind farm layout
        description satisfying the wind farm layout schema.
    kwargs : dict
        A (nested) `dict` containing a (partial) description
        of a wind farm layout according to the wind farm layout
        schema. It can be used both to override elements of
        the description given in `filename` or to define
        a wind farm layout in its entirety.
        
    Attributes
    ----------
    STATE_TYPES : frozenset
        The set of state types stored for layouts.
        Contains:

        * `'target'`: wake targets, i.e., turbines that contribute to
          power production (usually all the farm's turbines)
        * `'source'`: wake sources, i.e., turbines that generate wakes
          (usually farm turbines and all the surrounding turbines)
        * `'movable'`: turbines that are allowed to be moved
          (usually all the farm turbines, although some may be kept fixed
          at some point in the design)
        * `'context'`: turbines that belong to the surroundings
          and not to the farm that is being optimized
          (these should not be considered for constraint checks)

    Methods
    -------
    get_positions()
        Get positions.
    initialize_relative_positions(pos_from, pos_to)
        Add and initialize relative positions data in object.
    get_relative_positions()
        Get and, as needed, calculate relative positions.
    get_angles()
        Get and, as needed, calculate angles between locations.
    get_distances()
        Get and, as needed, calculate distances between locations.
    get_normed_relative_positions():
        Get and, as needed, calculate normed relative positions.
    set_positions(positions):
        Set the positions of the layout.
    shift_positions(step):
        Shift the positions of the layout by a given step.
        
    """
    STATE_TYPES = frozenset({'target', 'source', 'movable', 'context'})
    
    def __init__(self, filename=None, **kwargs):
        """Create a wind farm layout object."""
        # TODO: add reference to actual wind farm layout schema
        layout_dict = {}
        if filename is not None:
            with open(filename) as f:
                layout_dict.update(_hs.yaml_load(f))
        layout_dict.update(kwargs)
        # TODO: check layout_dict with schema using jsonschema
        
        layout = layout_dict['layout']
        if isinstance(layout, _xr.core.dataset.Dataset):
            ds = layout
        else:
            if isinstance(layout, list):
                layout_array = _np.array(layout_dict['layout'])
            elif isinstance(layout, _np.ndarray):
                layout_array = layout
            else:
                raise ValueError(f"Incorrect type of layout: {layout}.")
            ds = _xr.Dataset(coords={'pos': range(len(layout_array))})
            for name, index in {('x', 0), ('y', 1)}:
                ds[name] = ('pos', layout_array[:, index])
        self._state = ds
            
        # TODO: add other layout_dict properties as attributes to self?

    def get_positions(self):
        """Get positions.
        
        Returns
        -------
        `xarray.DataSet`
            ***
        
        """
        return self._state[['x', 'y']]

    def set_positions(self, positions):
        """Set the positions of the layout.
        
        Parameters
        ----------
        positions
            An `xarray.Dataset` with `x` and `y` `xarray.DataArray`s
            with a `pos` coordinate array containing a subset of
            positions present in the object's own `layout` attribute.
        
        """
        try:
            del self._rel  # rel becomes outdated when layout is changed
        except AttributeError:
            pass
        if not self._state.movable.sel(pos=positions.pos).all():
            raise IndexError("You may not change non-movable turbines.")
        for z in {'x', 'y'}:
            self._state[z].loc[dict(pos=positions.pos)] = positions[z]
            # TODO: do this in one go?

    def shift_positions(self, step):
        """Shift the positions of the layout by a given step.
        
        Parameters
        ----------
        step
            An `xarray.Dataset` with `x` and `y` `xarray.DataArray`s
            with a `pos` coordinate array containing a subset of
            positions present in the object's own `layout` attribute.
        
        """
        self.set_positions(self.positions.loc[dict(pos=step.pos)] + step)    
    
    def get_state(self, *args):
        """Get state information.
        
        Parameters
        ----------
        args
            ***
        
        Returns
        -------
        `xarray.Dataset`
            ***
        
        """
        illegal = set(args) - self.STATE_TYPES
        if illegal:
            raise ValueError(f"The arguments {illegal} are illegal types, "
                             f"i.e., not in {self.STATE_TYPES}.")
        undefined = set(args) - set(self._state.keys())
        if undefined:
            raise ValueError(f"The arguments {undefined} are not currently "
                             "defined for this layout. Use the ‘set_state’ "
                             "method for these first.")
        return self._state[list(args)]
    
    def set_state(self, **kwargs):
        """Set state information.
        
        Parameters
        ----------
        kwargs
            ***
        
        """
        args = kwargs.keys()
        if not self.STATE_TYPES.issuperset(set(args)):
            raise ValueError(f"The arguments given, {args}, includes illegal "
                             f"types, i.e., not in {self.STATE_TYPES}.")
        for state, val in kwargs.items():
            if isinstance(val, _xr.core.dataarray.DataArray):
                da = val
            elif isinstance(val, bool):
                da = _xr.full_like(self._state.pos, val, dtype=bool)
            else:
                raise ValueError(
                    f"Incorrect type of description for state {state}: {val}.")
            self._state[state] = da

    def initialize_relative_positions(self, pos_from, pos_to):
        """Add and initialize relative positions data in object.
        
        Parameters
        ----------
        pos_from
            Boolean array identifying positions
            to calculate relative positions from.
        pos_to
            Boolean array identifying positions
            to calculate relative positions to.
        
        """
        ds = self._state
        self._rel = _xr.Dataset(
            coords={'pos_from': ds.pos[pos_from].rename({'pos': 'pos_from'}),
                    'pos_to': ds.pos[pos_to].rename({'pos': 'pos_to'})}
        )

    def _has_rel_check(self):
        """Raise AttributeError if the _rel attribute is not present."""
        if not hasattr(self, '_rel'):
            raise AttributeError(
                "Call the ‘initialize_relative_positions’ method first.")

    def get_relative_positions(self):
        """Get and, as needed, calculate relative positions.
            
        Returns
        -------
        `xarray.DataSet`
            ***
        
        """
        self._has_rel_check()
        if ('x' not in self._rel) or ('y' not in self._rel):
            for z in {'x', 'y'}:
                da = self._state[z]
                self._rel[z] = (da.sel(pos=self._rel['pos_to'])
                                - da.sel(pos=self._rel['pos_from']))
                # TODO: do this in one go using self._state[['x', 'y']]?
        return self._rel[['x', 'y']]
    
    def get_angles(self):
        """Get and, as needed, calculate angles between positions.
            
        Returns
        -------
        `xarray.DataArray`
            ***
        
        """
        self._has_rel_check()
        if 'angle' not in self._rel:
            relxy = self.get_relative_positions()
            self._rel['angle'] = _np.arctan2(relxy.y, relxy.x)
        return self._rel.angle    

    def get_distances(self):
        """Get and, as needed, calculate distances between positions.
            
        Returns
        -------
        `xarray.DataArray`
            ***
        
        """
        self._has_rel_check()
        if 'distance' not in self._rel:
            relxy = self.get_relative_positions()
            self._rel['distance'] = _np.sqrt(
                _np.square(relxy.x) + _np.square(relxy.y))
        return self._rel.distance

    def get_normed_relative_positions(self):
        """Get and, as needed, calculate normed relative positions.
            
        Returns
        -------
        `xarray.DataArray`
            ***
        
        """
        self._has_rel_check()
        if ('x_normed' not in self._rel) or ('y_normed' not in self._rel):
            relxy = self.get_relative_positions()
            dists = self.get_distances()
            dists = dists.where(dists != 0, 1)  # avoid division by zero
            for z in {'x', 'y'}:
                self._rel[z + '_normed'] = relxy[z] / dists
                # TODO: do this in one go with relxy / dists?
        return self._rel[['x_normed', 'y_normed']]

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
