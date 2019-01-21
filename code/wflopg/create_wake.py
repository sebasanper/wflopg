import numpy as np
import xarray as xr


def _common(dc_vector_adim):
    downwind = dc_vector_adim.sel(dc_coord='d', drop=True)
    crosswind = np.abs(dc_vector_adim.sel(dc_coord='c', drop=True))
    is_downwind = downwind > 0
    return downwind, crosswind, is_downwind


def _half_lens_area(distance, own_radius, other_radius, mask):
    """Return relative area of half-lens

    https://christianhill.co.uk/blog/overlapping-circles/

    This function is used to calculate relative areas for analytical partial
    wake calculations. It is relevant for models with top-hat wake profiles.

    """
    cosine = xr.where(
        mask,
        (distance ** 2 + own_radius ** 2 - other_radius ** 2)
        / (2 * distance * own_radius),
        np.nan
    )
    angle = xr.where(mask, np.arccos(cosine), np.nan)
    return xr.where(
        mask,
        (angle - np.sin(2 * angle) / 2) * own_radius ** 2 / np.pi,
        np.nan
    )


def rss_combination():
    """Return the root-sum-square wake deficit combination rule"""
    def combination_rule(deficit):
        """Return combined and relative wake deficits

        deficit must be an xarray DataArray of individual deficits with
        'source' as one dimension.

        """
        squared = np.square(deficit)
        squared_combined = squared.sum(dim='source')
        relative = xr.where(
            squared_combined > 0, squared / squared_combined, 0)
        squared_combined_saturated = xr.where(  # RSS does not guarantee <= 1
            squared_combined <= 1, squared_combined, 1)
        return np.sqrt(squared_combined_saturated), relative

    return combination_rule


def bpa_iea37(thrust_curve, rotor_radius, turbulence_intensity):
    """Return an IEA37-variant Bastankhah–Porté-Agel wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit at the turbine hub is used, so there is no rotor
    plane averaging.

    """
    expansion_coeff = 0.3837 * turbulence_intensity + 0.003678
    sigma_at_source = 1 / np.sqrt(2)

    def wake_model(dc_vector):
        """Return wind speed deficit due to wake

        The argument must be an xarray DataArray of dimensional
        downwind-crosswind coordinate pairs.

        """
        downwind, crosswind, is_downwind = _common(dc_vector / rotor_radius)
        sigma = xr.where(
            is_downwind, sigma_at_source + expansion_coeff * downwind, np.nan)
        exponent = xr.where(is_downwind, -(crosswind / sigma) ** 2 / 2, np.nan)
        radical = xr.where(
            is_downwind, 1 - thrust_curve / (2 * sigma ** 2), np.nan)
        return xr.where(
            is_downwind, (1. - np.sqrt(radical)) * np.exp(exponent), 0
        ).transpose('direction', 'wind_speed', 'source', 'target')

    return wake_model


def _jensen_generic(thrust_curve, rotor_radius, expansion_coeff,
                   frandsen=False, averaging=False):
    """Return a Jensen wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    """
    induction_factor = 1 - np.sqrt(1 - thrust_curve)
    if frandsen:  # use (adim.) stream tube radius instead or rotor radius
        stream_tube_radius = np.sqrt((1 - induction_factor / 2)
                                     / (1 - induction_factor))
    else:
        stream_tube_radius = 1

    def wake_model(dc_vector):
        """Return wind speed deficit due to wake

        The argument must be an xarray DataArray of dimensional
        downwind-crosswind coordinate pairs.

        """
        downwind, crosswind, is_downwind = _common(dc_vector / rotor_radius)
        wake_radius = xr.where(
            is_downwind,
            1 + expansion_coeff * downwind / stream_tube_radius,
            np.nan)
        waked = is_downwind & (crosswind < 1 + wake_radius)
        if averaging:
            partially = waked & (1 + crosswind > wake_radius)
            relative_area = xr.where(
                waked,
                xr.where(
                    partially,
                    _half_lens_area(crosswind, 1, wake_radius, partially) +
                    _half_lens_area(crosswind, wake_radius, 1, partially),
                    1
                ),
                0
            )
        else:
            relative_area = 1
        return xr.where(
            waked,
            relative_area * induction_factor / np.square(wake_radius),
            0
        ).transpose('direction', 'wind_speed', 'source', 'target')

    return wake_model


def jensen(thrust_curve, rotor_radius, expansion_coeff):
    """Return a Jensen wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit at the turbine hub is used, so there is no rotor
    plane averaging.

    """
    return _jensen_generic(thrust_curve, rotor_radius, expansion_coeff)


def jensen_frandsen(thrust_curve, rotor_radius, expansion_coeff):
    """Return a Jensen wake model function as defined by Frandsen

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit at the turbine hub is used, so there is no rotor
    plane averaging. Frandsen's modification is taking the stream tube radius
    downwind of the rotor instead of the rotor radius itself.

    """
    return _jensen_generic(
        thrust_curve, rotor_radius, expansion_coeff, frandsen=True)


def jensen_averaged(thrust_curve, rotor_radius, expansion_coeff):
    """Return a Jensen partial wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit is averaged over the rotor plane, so partial wakes
    are used.

    """
    return _jensen_generic(
        thrust_curve, rotor_radius, expansion_coeff, averaging=True)


def jensen_frandsen_averaged(thrust_curve, rotor_radius, expansion_coeff):
    """Return a Jensen partial wake model function as defined by Frandsen

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit is averaged over the rotor plane, so partial wakes
    are used. Frandsen's modification is taking the stream tube radius downwind
    of the rotor instead of the rotor radius itself.

    """
    return _jensen_generic(thrust_curve, rotor_radius, expansion_coeff,
                           frandsen=True, averaging=True)

