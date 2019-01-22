import numpy as np
import xarray as xr


def generate_downwind(directions):
    """Return downwind unit vectors for all given directions

    directions must be an xarray DataArray of directions in degrees,
    where 0° is North and 90° is East.

    """
    # Convert inflow wind direction
    # - from windrose (N=0, CW) to standard (E=0, CCW): 90 - wind_dir
    # - from upwind to downwind: +180
    # - from degrees to radians
    directions_rad = np.radians(90 - directions + 180)
    return xr.concat(
        [np.cos(directions_rad), np.sin(directions_rad)], 'xy'
    ).transpose()  # transpose to get direction as first dimension


def generate_crosswind(downwind):
    """Return crosswind vectors corresponding to the given downwind vectors

    downwind must be an xarray DataArray with xy as one dimension.
    The crosswind vectors are normal to the downwind vectors and form a
    right-hand coordinate system (positive rotation is counterclockwise).

    """
    crosswind = downwind.roll(xy=1)
    crosswind.coords['xy'] = downwind.coords['xy']
        # workaround for bug in roll that also rolls coordinates;
        # fixed in xarray 0.10.9
    return np.negative(crosswind, out=crosswind.values, where=[True, False])
        # out is needed because of https://stackoverflow.com/questions/54250461


def generate_vector(context, layout):
    """Return the vectors from context turbines to layout turbines

    The arguments must be xarray DataArrays that have an xy dimension.
    Such a DataArray is also returned.

    """
    return (layout - context).transpose('source', 'target', 'xy')


def generate_distance(vector):
    """Return the distances between context and layout turbines

    The argument must be an xarray DataArray that has an xy dimension.
    A DataArray with the same dimensions, but omitting the xy
    dimension, is returned.

    """
    return np.sqrt(np.square(vector).sum(dim='xy'))


def generate_dc_vector(vector, downwind, crosswind):
    """Return inter-turbine vectors in downwind-crosswind coordinates

    The first argument must be an xarray DataArray that has an xy
    dimension. It is assumed to represent vectors between turbines in a
    classical coordinate system. The second and third arguments must be xarray
    DataArrays of downwind and crosswind vectors, respectively.

    """
    return xr.concat(
        [vector.dot(downwind), vector.dot(crosswind)], 'dc'
    ).transpose('direction', 'source', 'target', 'dc')
