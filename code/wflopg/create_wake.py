import numpy as np
import xarray as xr


def _half_lens_area(distance, own_radius, other_radius):
    """Return relative area of half-lens

    https://christianhill.co.uk/blog/overlapping-circles/

    This function is used to calculate relative areas for analytical partial
    wake calculations.

    """
    angle = np.acos((distance ** 2 + own_radius ** 2 - other_radius ** 2)
                    / (2 * distance * own_radius))
    return (angle - np.sin(2 * angle) / 2) / np.pi


def rss_combination():
    """Return the root-mean-square wake deficit combination rule"""
    def combination_rule(deficit):
        """Return combined wake deficits

        deficit must be an xarray DataArray of individual deficits with
        'source' as one dimension.

        """
        return np.sqrt(np.square(deficit).sum(dim='source'))

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

    def wake_model(downstream):
        """Return wind speed deficit due to wake

        The argument must be an xarray DataArray of dimensional
        downwind-crosswind coordinate pairs.

        """
        downstream /= rotor_radius # make adimensional
        downwind = downstream.sel(dc_coord='d') > 0
        sigma = xr.where(
            downwind,
            sigma_at_source + expansion_coeff * downstream.sel(dc_coord='d'),
            np.nan
        )
        exponent = xr.where(
            downwind,
            -(downstream.sel(dc_coord='c') / sigma) ** 2 / 2,
            np.inf
        )
        radical = xr.where(
            downwind, 1 - thrust_curve / (2 * sigma ** 2), np.nan)
        return xr.where(
            downwind, (1. - np.sqrt(radical)) * np.exp(exponent), 0)

    return wake_model


def _jensen_generic(thrust_curve, rotor_radius, hub_height, surface_roughness,
                   frandsen=False, averaging=False):
    """Return a Jensen wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    """
    # NOTE: alternative, constant values for the expansion coefficient
    #       available in the literature are 0.075 for onshore
    #       and 0.0–4.05 for offshore
    #
    expansion_coeff = 0.5 / np.log(hub_height / surface_roughness)
    induction_factor = 1 - np.sqrt(1 - thrust_curve)
    if frandsen:  # use (adim.) stream tube radius instead or rotor radius
        stream_tube_radius = np.sqrt((1 - induction_factor / 2)
                                     / (1 - induction_factor))
    else:
        stream_tube_radius = 1

    def wake_model(downstream):
        """Return wind speed deficit due to wake

        The argument must be an xarray DataArray of dimensional
        downwind-crosswind coordinate pairs.

        """
        downstream /= rotor_radius # make adimensional
        downwind = downstream.sel(dc_coord='d') > 0
        wake_radius = xr.where(
            downwind,
            (1 + expansion_coeff * downstream.sel(dc_coord='d')
                                                         / stream_tube_radius),
            np.nan)
        waked = xr.where(
            downwind,
            downstream.sel(dc_coord='c') < wake_radius,
            False
        )
        if averaging:
            partially = xr.where(
                waked, 1 + downstream.sel(dc_coord='c') < wake_radius, False)
            relative_area = xr.where(
                waked,
                xr.where(
                    partially,
                    _half_lens_area(
                        downstream.sel(dc_coord='c'), 1, wake_radius) +
                    _half_lens_area(
                        downstream.sel(dc_coord='c'), wake_radius, 1),
                    1
                ),
                0
            )
        else:
            relative_area = 1
        return xr.where(
            waked,
            induction_factor / np.square(relative_area * wake_radius),
            0
        )

    return wake_model


def jensen(thrust_curve, rotor_radius, hub_height, surface_roughness):
    """Return a Jensen wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit at the turbine hub is used, so there is no rotor
    plane averaging.

    """
    return _jensen_generic(
        thrust_curve, rotor_radius, hub_height, surface_roughness)


def jensen_frandsen(thrust_curve, rotor_radius, hub_height, surface_roughness):
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
        thrust_curve, rotor_radius, hub_height, surface_roughness,
        frandsen=True
    )


def jensen_averaged(thrust_curve, rotor_radius, hub_height, surface_roughness):
    """Return a Jensen partial wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit is averaged over the rotor plane, so partial wakes
    are used.

    """
    return _jensen_generic(
        thrust_curve, rotor_radius, hub_height, surface_roughness,
        averaging=True
    )


def jensen_frandsen_averaged(
        thrust_curve, rotor_radius, hub_height, surface_roughness):
    """Return a Jensen partial wake model function as defined by Frandsen

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    The value of the deficit is averaged over the rotor plane, so partial wakes
    are used. Frandsen's modification is taking the stream tube radius downwind
    of the rotor instead of the rotor radius itself.

    """
    return _jensen_generic(
        thrust_curve, rotor_radius, hub_height, surface_roughness,
        frandsen=True, averaging=True
    )

