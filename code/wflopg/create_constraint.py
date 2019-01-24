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


def outside_parcels(parcels, layout, safety_distance):
    """Check which turbines in the layout are outside the parcels

    parcels is a list of nested dicts of constraints and exclusions, where the
    constraints are formulated as linear expressions that evaluate to a
    positive number on their ‘outside’ side or as circles where ‘outside’
    corresponds to outside the delimited disc.

    layout is an xarray DataArray of xy-coordinates for the turbines.

    safety_distance should be the site-adimensional rotor radius

    """
    layout_mon = _xy_to_monomial(layout)

    def outside_parcels_for_monomial(parcels, outside, exclusion=False):
        """Return which turbines are outside the given parcels

        This recursive function walks over all parcels and their exclusions.
        Whether or not it is currently operating in an exclusion is tracked by
        the exclusion variable. The xarray DataArray outside is a boolean array
        with the same shape of the layout, but removing the xy-dimension.

        """
        # TODO: we should apply an xr.where based on already violating at
        #       higher levels
        for parcel in parcels:
            if 'constraints' in parcel:
                rotor_constraint = parcel['constraints'].rotor_constraint
                # turbines with a positive constraint evaluation value violate
                # that constraint
                violates = parcel['constraints'].dot(layout_mon) > 0
                    # TODO: * we pretend for now that the LHS is the distance
                    #       * the safety distance needs to be added/subtracted
                    #         depending on the exclusion state and rotor
                    #         constraint requirement
                    #       * pre-compute some variables
                if exclusion:
                    # in an exclusion, only if all constraints are violated is
                    # the turbine actually outside the area defined by the
                    # constraints
                    outside |= violates.all(dim='constraint')
                else:
                    # otherwise, the turbine is outside the area if any of the
                    # constraints is violated
                    outside |= violates.any(dim='constraint')
            elif 'circle' in parcel:
                center = parcel['circle']
                radius = parcel['circle'].radius
                dist = radius
                if parcel['circle'].rotor_constraint:
                    dist += safety_distance if exclusion else -safety_distance
                # TODO: pre-compute np.square(dist)!
                in_disc = (np.square(layout - center).sum(dim='xy')
                           <= np.square(dist))
                outside |= in_disc if exclusion else ~in_disc
            if 'exclusions' in parcel:
                # recurse to evaluate an exclusion (which may be an inclusion
                # if its inside an exclusion, so we flip the exclusion
                # variable's truth value)
                outside |= outside_parcels_for_monomial(
                    parcel['exclusions'], outside, not exclusion)
        return outside

    # start DataArray defined by trivial test
    outside = np.square(layout).sum(dim='xy') > 1
    return outside_parcels_for_monomial(parcels, outside)
