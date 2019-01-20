import numpy as np
import xarray as xr


def distance(turbine_distance, rotor_radius, site_radius):
    """Return a generator of steps that can fix turbine distance constraints

    The turbine_distance is assumed to be in rotor diameters.

    """
    threshold = turbine_distance * (2 * rotor_radius) / site_radius

    def proximity_repulsion(vector, distance):
      """Return steps that can fix turbine constraints

      Both vector and distance must be DataArrays with source and
      target as dimensions, of adimensional xy-coordinate pairs and distances,
      respectively.

      """
      violation = (0 < distance) & (distance < threshold)
          # NOTE: 0 excluded for distance-to-self,
          #       so try to avoid turbines at the same location
      if np.any(violation):
          repulsion_step = xr.where(
              violation,
              (threshold - distance) / 2 * vector,  # just enough to fix issue
              [0, 0]
          )
          return repulsion_step.sum(dim='source')
      else:
          return None

    return proximity_repulsion
