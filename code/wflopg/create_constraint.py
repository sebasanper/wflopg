import numpy as np
import xarray as xr

from wflopg.constants import COORDS


def _xy_to_quad(xy):
    """Return quadratic ‘coordinates’ for the given xy-coordinates

    Quadratic coordinates are, in the following order,

      1, x, y, x⋅y, x² (xx), and y² (yy).

    This function works for any xarray DataArray with xy as a dimension.

    """
    x = xy.sel(xy='x', drop=True)
    y = xy.sel(xy='y', drop=True)
    one = x.copy()
    one.values = np.ones(one.shape)
    quad = xr.concat(
        [one, x, y, x*y, np.square(x), np.square(y)], 'quad').transpose()
    quad.coords['quad'] = COORDS['quad']
    return quad


def distance(turbine_distance):
    """Return a generator of steps that can fix turbine distance constraints

    The turbine_distance is assumed to be site-distance adimensional.

    """
    def proximity_repulsion(vector, distance):
      """Return steps that can fix turbine constraints

      Both vector and distance must be DataArrays with source and
      target as dimensions, of adimensional xy-coordinate pairs and distances,
      respectively.

      """
      violation = (0 < distance) & (distance < turbine_distance)
          # NOTE: 0 excluded for distance-to-self,
          #       so try to avoid turbines at the same location
      if np.any(violation):
          repulsion_step = xr.where(
              violation,
              (turbine_distance - distance) / 2 * vector,
                  # just enough to fix the issue
              [0, 0]
          )
          return repulsion_step.sum(dim='source')
      else:
          return None

    return proximity_repulsion


def outside_parcels(parcels, layout, safety_distance=0):
    """Check which turbines in the layout are outside the parcels

    parcels is a list of nested dicts of constraints and exclusions, where the
    constraints are formulated as quadratic expressions that evaluate to a
    positive number on their ‘outside’ side.

    layout is an xarray DataArray of xy-coordinates for the turbines.

    safety distance is the minimum distance the turbines need to be placed
    inside the parcels.

    """
    layout_quad = _xy_to_quad(layout)

    def outside_parcels_for_quad(parcels, outside, exclusion=False):
        """Return which turbines are outside the given parcels

        This recursive function walks over all parcels and their exclusions.
        Whether or not it is currently operating in an exclusion is tracked by
        the exclusion variable. The xarray DataArray outside is a boolean array
        with the same shape of the layout, but removing the xy-dimension.

        """
        for parcel in parcels:
            if 'constraints' in parcel:
                # turbines with a positive constraint evaluation value violate
                # that constraint
                violates = parcel['constraints'].dot(layout_quad) > 0  # TODO: we pretend for now that the LHS is the distance
                if exclusion:
                    # in an exclusion, only if all constraints are violated is
                    # the turbine actually outside the area defined by the
                    # constraints
                    outside |= violates.all(dim='constraint')
                else:
                    # otherwise, the turbine is outside the area if any of the
                    # constraints is violated
                    outside |= violates.any(dim='constraint')
            if 'exclusions' in parcel:
                # recurse to evaluate an exclusion (which may be an inclusion
                # if its inside an exclusion, so we flip the exclusion
                # variable's truth value)
                outside |= outside_parcels_for_quad(
                    parcel['exclusions'], outside, not exclusion)
        return outside

    outside = (layout > 1).any(dim='xy')  # start DataArray with trivial test
    return outside_parcels_for_quad(parcels, outside)
