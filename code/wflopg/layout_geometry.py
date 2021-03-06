"""Functions generating information about the layout geometry."""

import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS
from wflopg.helpers import rss


def generate_downwind(directions):
    """Return downwind unit vectors for all given directions

    directions must be an xarray DataArray of directions in degrees,
    where 0° is North and 90° is East.

    """
    # Convert inflow wind direction
    # - from windrose (N=0, CW) to standard (E=0, CCW): 90 - wind_dir
    # - from upwind to downwind: +180
    # - from degrees to radians
    directions_rad = _np.radians(90 - directions + 180)
    return _xr.concat([_np.cos(directions_rad), _np.sin(directions_rad)], 'xy')


def generate_crosswind(downwind):
    """Return crosswind vectors corresponding to the given downwind vectors

    downwind must be an xarray DataArray with xy as one dimension.
    The crosswind vectors are normal to the downwind vectors and form a
    right-hand coordinate system (positive rotation is counterclockwise).

    """
    return (downwind.roll(xy=1, roll_coords=False)
            * _xr.DataArray([-1, 1], coords=[('xy', COORDS['xy'])]))


def generate_vector(context, layout):
    """Return the vectors from context turbines to layout turbines

    The arguments must be xarray DataArrays that have an xy dimension.
    Such a DataArray is also returned.

    """
    return layout - context


def generate_distance(vector):
    """Return the distances between context and layout turbines

    The argument must be an xarray DataArray that has an xy dimension.
    A DataArray with the same dimensions, but omitting the xy
    dimension, is returned.

    """
    return rss(vector, dim='xy')


def generate_dc_vector(vector, downwind, crosswind):
    """Return inter-turbine vectors in downwind-crosswind coordinates

    The first argument must be an xarray DataArray that has an xy
    dimension. It is assumed to represent vectors between turbines in a
    classical coordinate system. The second and third arguments must be xarray
    DataArrays of downwind and crosswind vectors, respectively.

    """
    return _xr.concat([vector.dot(downwind), vector.dot(crosswind)], 'dc')
