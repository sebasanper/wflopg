"""Functions that generate functions modeling a turbine."""

import xarray as _xr


def _check_start(interpolation_data, start_speed, start_value):
    """Adapt interpolation_data or start_speed as needed

    The idea is that if start_speed lies below the smallest interpolation_data
    coordinate value (assumed ordered), then an extra interpolating point
    should be added. Otherwise start_speed should be changed to the smallest
    interpolation_data coordinate value.

    """
    min_interpolation_speed = interpolation_data.speed.min().item()
    if start_speed < min_interpolation_speed:
        interpolation_data = _xr.concat(
            [_xr.DataArray([start_value], coords=[('speed', [start_speed])]),
             interpolation_data],
            'speed'
        )
    else:
        start_speed = min_interpolation_speed
    return interpolation_data, start_speed


def _check_end(interpolation_data, end_speed, end_value):
    """Adapt interpolation_data or end_speed as needed

    The idea is that if end_speed lies below the largest interpolation_data
    coordinate value (assumed ordered), then an extra interpolating point
    should be added. Otherwise end_speed should be changed to the largest
    interpolation_data coordinate value.

    """
    max_interpolation_speed = interpolation_data.speed.max().item()
    if end_speed > max_interpolation_speed:
        interpolation_data = _xr.concat(
            [interpolation_data,
             _xr.DataArray([end_value], coords=[('speed', [end_speed])])],
            'speed'
        )
    else:
        end_speed = max_interpolation_speed
    return interpolation_data, end_speed


def _within_cut(speeds, cut_in, cut_out):
    return (speeds >= cut_in) & (speeds <= cut_out)  # within cut


def cubic_power_curve(rated_power, rated_speed, cut_in, cut_out):
    """Return a cubic power curve function"""
    def power_curve(speeds):
        """Return turbine power for the given (wind) speeds

        speeds is assumed to be an xarray DataArray.

        """
        wc = _within_cut(speeds, cut_in, cut_out)
        return rated_power * (
            wc
            * (speeds.where(speeds < rated_speed, rated_speed) - cut_in)
            / (rated_speed - cut_in)
        ) ** 3

    return power_curve


def interpolated_power_curve(rated_power, rated_speed, cut_in, cut_out,
                             interpolation_data):
    """Return an interpolated power curve function

    Interpolation data, a one-dimensional DataArray with 'speed' coordinate
    (assumed ordered)is assumed to be authorative over other specified
    parameters.

    """
    interpolation_data, cut_in = _check_start(interpolation_data, cut_in, 0)
    interpolation_data, end_speed = _check_end(interpolation_data,
                                               rated_speed, rated_power)
    interpolation_data, cut_out = _check_end(interpolation_data,
                                             cut_out, rated_power)

    def power_curve(speeds):
        """Return turbine power for the given (wind) speeds

        speeds is assumed to be an xarray DataArray.

        """
        return interpolation_data.interp(
            speed=speeds, kwargs={'fill_value': 0.0}).drop('speed')

    return power_curve


def constant_thrust_curve(cut_in, cut_out, thrust_coefficient):
    """Return a constant thrust curve function"""
    def thrust_curve(speeds):
        """Return turbine thrust for the given (wind) speeds

        speeds is assumed to be an xarray DataArray.

        """
        wc = _within_cut(speeds, cut_in, cut_out)  # within cut
        return wc * thrust_coefficient

    return thrust_curve


def interpolated_thrust_curve(cut_in, cut_out, interpolation_data):
    """Return an interpolated thrust curve function

    Interpolation data, a one-dimensional DataArray with 'speed' coordinate
    (assumed ordered), is assumed to be authorative over other specified
    parameters.

    """
    interpolation_data, cut_in = _check_start(interpolation_data, cut_in, 0.)
    interpolation_data, cut_out = _check_end(interpolation_data, cut_out, 0.)

    def thrust_curve(speeds):
        """Return turbine thrust for the given (wind) speeds

        speeds is assumed to be an xarray DataArray.

        """
        # only 1D-arrays can be interpolated
        return interpolation_data.interp(
            speed=speeds, kwargs={'fill_value': 0.0}).drop('speed')

    return thrust_curve
