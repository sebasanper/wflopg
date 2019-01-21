import numpy as np
import xarray as xr


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
