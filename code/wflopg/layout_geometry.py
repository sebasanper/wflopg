import numpy as np
import xarray as xr


def generate_vector(context, layout):
    """Return the vectors from context turbines to layout turbines

    The arguments must be xarray DataArrays that have an xy_coord dimension.
    Such a DataArray is also returned.

    """
    return (layout - context).transpose('source', 'target', 'xy_coord')


def generate_distance(vector):
    """Return the distances between context and layout turbines

    The argument must be an xarray DataArray that has an xy_coord dimension.
    A DataArray with the same dimensions, but omitting the xy_coord
    dimension, is returned.

    """
    return np.sqrt(np.square(vector).sum(dim='xy_coord'))


def generate_downstream(vector, downwind):
    """Return inter-turbine vectors in downwind-crosswind coordinates

    The first argument must be an xarray DataArray that has an xy_coord
    dimension. It is assumed to represent vectors between turbines in a
    classical coordinate system. The second argument must be a an xarray
    DataArray of downwind vectors.

    """
    crosswind = np.negative(downwind.roll(xy_coord=1), where=[True, False])
    crosswind.coords['xy_coord'] = downwind.coords['xy_coord']  # workaround for bug in roll that also rolls coordinates; fixed in xarray 0.10.9
    return xr.concat(
        [vector.dot(downwind), vector.dot(crosswind)], 'dc_coord'
    ).transpose('direction', 'source', 'target', 'dc_coord')
