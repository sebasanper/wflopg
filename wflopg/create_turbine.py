import numpy as np
import xarray as xr


def _check_start(interpolation_data, start_speed, start_value):
    """Adapt interpolation_data or start_speed as needed

    The idea is that if start_speed lies below the smallest interpolation_data
    coordinate value (assumed ordered), then an extra interpolating point
    should be added. Otherwise start_speed should be changed to the smallest
    interpolation_data coordinate value.

    """
    if start_speed < interpolation_data[0, 0]:
        interpolation_data = np.concatenate(([[start_speed, start_value]],
                                             interpolation_data))
    else:
        start_speed = interpolation_data[0, 0]
    return interpolation_data, start_speed


def _check_end(interpolation_data, end_speed, end_value):
    """Adapt interpolation_data or end_speed as needed

    The idea is that if end_speed lies below the largest interpolation_data
    coordinate value (assumed ordered), then an extra interpolating point
    should be added. Otherwise end_speed should be changed to the largest
    interpolation_data coordinate value.

    """
    if end_speed > interpolation_data[-1, 0]:
        interpolation_data = np.concatenate((interpolation_data,
                                             [[end_speed, end_value]]))
    else:
        end_speed = interpolation_data[-1, 0]
    return interpolation_data, end_speed


def _create_interpolator(coord_name, interpolation_data):
    """Put interpolation_data into an xarray DataArray

    The DataArray supplies interpolation functionality. It is assumed that
    interpolation data is a two-column numpy array in which the first column
    contains the coordinate values and the second the function values to be
    interpolated.

    """
    return xr.DataArray(interpolation_data[:, 1],
                                      [(coord_name, interpolation_data[:, 0])])


def _common(speeds, cut_in, cut_out):
    if np.any(speeds < 0):
        raise ValueError("Wind speeds may not be negative.")
    powers = np.zeros(speeds.shape)
    wc = speeds > cut_in & speeds < cut_out  # within cut
    return out_array, wc


def cubic_power_curve(rated_power, rated_speed, cut_in, cut_out):
    """Return a cubic power curve function"""
    def power_curve(speeds):
        """Return turbine output powers for the given (wind) speeds

        speeds is assumed to be a numpy array.

        """
        powers, wc = _common(speeds, cut_in, cut_out)
        br = speeds < rated_speed  # below rated
        powers[wc] = rated_power
        powers[wc & br] = ((speeds[wc & br] - cut_in)
                           / (rated_speed - cut_in)) ** 3
        return powers

    return power_curve


def interpolated_power_curve(rated_power, rated_speed, cut_in, cut_out,
                                    interpolation_data):
    """Return an interpolated power curve function

    Interpolation data is assumed to be authorative over other specified
    parameters.

    """
    interpolation_data = interpolation_data[interpolation_data.argsort(axis=0)]
    interpolation_data, cut_in = _check_start(interpolation_data, cut_in, 0)
    interpolation_data, end_speed = _check_end(interpolation_data,
                                               rated_speed, rated_power)
    interpolation_data, cut_out = _check_end(interpolation_data,
                                             cut_out, rated_power)
    interpolator = _create_interpolator('speeds', interpolation_data)

    def power_curve(speeds):
        """Return turbine output powers for the given (wind) speeds

        speeds is assumed to be a numpy array.

        """
        powers, wc = _common(speeds, cut_in, cut_out)
        powers[wc] = interpolator.interp(speeds=speeds[wc]).values
        return powers

    return power_curve


def constant_thrust_curve(cut_in, cut_out, thrust_coefficient):
    """Return a constant thrust curve function"""
    def thrust_curve(speeds):
        """Return turbine output thrust for the given (wind) speeds

        speeds is assumed to be a numpy array.

        """
        thrusts, wc = _common(speeds, cut_in, cut_out)
        thrusts[wc] = thrust_coefficient
        return thrusts

    return thrust_curve


def interpolated_thrust_curve(cut_in, cut_out, interpolation_data):
    """Return an interpolated thrust curve function

    Interpolation data is assumed to be authorative over other specified
    parameters.

    """
    interpolation_data = interpolation_data[interpolation_data.argsort(axis=0)]
    interpolation_data, cut_in = _check_start(interpolation_data, cut_in, 0.)
    interpolation_data, cut_out = _check_end(interpolation_data, cut_out, 0.)
    interpolator = _create_interpolator('speeds', interpolation_data)

    def thrust_curve(speeds):
        """Return turbine output thrust for the given (wind) speeds

        speeds is assumed to be a numpy array.

        """
        thrusts, wc = _common(speeds, cut_in, cut_out)
        thrusts[wc] = interpolator.interp(speeds=speeds[wc]).values
        return thrusts

    return thrust_curve
