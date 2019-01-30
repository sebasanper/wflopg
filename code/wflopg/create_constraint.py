import numpy as np
import xarray as xr

from wflopg.constants import COORDS


def _xy_to_monomial(xy):
    """Return monomial ‘coordinates’ for the given xy-coordinates

    This function works for any xarray DataArray with xy as a dimension.

    """
    x = xy.sel(xy='x', drop=True)
    y = xy.sel(xy='y', drop=True)
    one = xr.ones_like(x)
    mon = xr.concat([one, x, y], 'monomial').transpose()
    mon.coords['monomial'] = COORDS['monomial']
    return mon


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


def inside_parcels(parcels, layout):
    """Check which turbines in the layout are outside the parcels

    parcels is a list of nested dicts of constraints and exclusions, where the
    constraints are formulated as linear expressions that evaluate to a
    positive number on their ‘outside’ side or as circles where ‘outside’
    corresponds to outside the delimited disc.

    layout is an xarray DataArray of xy-coordinates for the turbines.

    """
    layout_mon = _xy_to_monomial(layout)

    def inside_recursive(parcel, undecided, exclusion=False):
        """Return which turbines are inside the given parcel

        This recursive function walks over the parcel and its exclusions.
        Whether or not it is currently operating in an exclusion is tracked by
        the exclusion variable. The xarray DataArray undecided is a boolean
        array with the same shape of the layout, but removing the xy-dimension.
        It contains the turbines to consider.

        """
        ##
        if 'constraints' in parcel:
            distance = xr.where(  # signed distance
                undecided, parcel['constraints'].dot(layout_mon), np.nan)
            # turbines with a nonpositive constraint evaluation value
            # satisfy that constraint
            satisfies = xr.where(undecided, distance <= 0, False)
            if exclusion:
                # if no (not any) constraint is satisfied, then the
                # turbine is excluded (at this recursion level)
                outside = ~satisfies.any(dim='constraint')
            else:
                # excluded turbines are inside the parcel if all of the
                # constraints at this deeper recursion level are satisfied
                included = satisfies.all(dim='constraint')
        elif 'circle' in parcel:
            in_disc = xr.where(
                undecided,
                np.square(layout - parcel['circle']).sum(dim='xy')
                <= parcel['circle'].dist_sqr,
                False
            )
            if exclusion:
                outside = ~in_area
            else:
                included = in_area
        ##
        if exclusion:
            inside = undecided.copy()
            undecided &= outside
        else:
            inside = undecided & included
            undecided = inside

        if 'exclusions' in parcel:
            # recurse to evaluate an exclusion (which may be an inclusion
            # if its inside an exclusion, so we flip the exclusion
            # variable's truth value)
            inside = xr.where(
                undecided,
                parcel_walker(parcel['exclusions'], undecided, not exclusion),
                inside
            )
        else:  # end of recursion
            if exclusion:
                inside = xr.where(undecided, ~undecided, inside)

        return inside


    def parcel_walker(parcels, undecided, exclusion=False):
        return xr.concat(
            [inside_recursive(parcel, undecided, exclusion)
             for parcel in parcels],
            'parcels'
        ).all(dim='parcels')


    # start DataArray defined by trivial test
    inside = np.square(layout).sum(dim='xy') <= 1
    return parcel_walker(parcels, inside)
