import numpy as _np
import xarray as _xr


def rss(array, dim):
    """Calculate root-sum-square of array along dim."""
    return _np.sqrt(_np.square(array).sum(dim=dim))


def cyclic_extension(array, dim, coord_val=0, add=True):
    """Cyclicly extend an array

    This function extends the given xarray DataArray along the given dimension
    by appending the first value at the end. That dimension must have an
    associated coordinate. The coordinate value at that new last position is
    changed using the coordinate value given. By default, this value is added
    to the original coordinate value, but one can choose to replace it.

    """
    coord = array.coords[dim]
    coord_cyc = _xr.concat([coord, coord[0]], dim)
    array_cyc = array.sel({dim: coord_cyc})
    if add:
        coord_cyc[-1] += coord_val
    else:
        coord_cyc[-1] = coord_val
    array_cyc.coords[dim] = coord_cyc
    return array_cyc
