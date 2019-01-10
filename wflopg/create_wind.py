import numpy as np


def logarithmic_wind_shear(reference_height, roughness_length):
    """Return a logarithmic wind shear function"""
    def wind_shear(heights, speeds):
        """Return sheared wind speed at the given heights

        heights and speeds are assumed to be numpy arrays of the same size.

        """
        return speeds * np.log((heights - reference_height) / roughness_length)
