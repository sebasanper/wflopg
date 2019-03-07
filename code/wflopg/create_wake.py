import numpy as np
import xarray as xr


def _common(dc_vector_adim):
    downwind = dc_vector_adim.sel(dc='d', drop=True)
    crosswind = np.abs(dc_vector_adim.sel(dc='c', drop=True))
    is_downwind = downwind > 0
    return downwind, crosswind, is_downwind


def _lens_area(d, r, R):
    """Return relative area of half-lens

    This function is used to calculate the area of a lens with radii r and R at
    distance d.

    From <http://mathworld.wolfram.com/Lens.html>, modified for efficiency.

    (Alternative: <https://christianhill.co.uk/blog/overlapping-circles/>)

    """
    dd = np.square(d)
    rr = np.square(r)
    RR = np.square(R)
    RR_rr = RR - rr
    return (
        rr * np.arccos((dd - RR_rr) / (2 * d * r))
        +
        RR * np.arccos((dd + RR_rr) / (2 * d * R))
        -
        np.sqrt(2 * dd * (RR + rr) - np.square(dd) - np.square(RR_rr)) / 2
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
        relative = squared / (squared_combined + (squared_combined == 0))
            # we add 1 where the denominator (and numerator) would be 0
        squared_combined_saturated = np.minimum(squared_combined, 1)
            # RSS does not guarantee <= 1
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
        sigma = sigma_at_source + expansion_coeff * downwind * is_downwind
            # multiplication with is_downwind to avoid negative radical later
        exponent = -(crosswind / sigma) ** 2 / 2
        radical = 1 - thrust_curve / (2 * sigma ** 2)
        return is_downwind * (1. - np.sqrt(radical)) * np.exp(exponent)

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
            # TODO: in case thrust_curve == 0, we get divide by zero!
    else:
        stream_tube_radius = 1

    def wake_model(dc_vector):
        """Return wind speed deficit due to wake

        The argument must be an xarray DataArray of dimensional
        downwind-crosswind coordinate pairs.

        """
        downwind, crosswind, is_downwind = _common(dc_vector / rotor_radius)
        wake_radius = 1 + expansion_coeff * downwind / stream_tube_radius
        if averaging:
            waked = is_downwind & (crosswind < 1 + wake_radius)
            relative_area = np.float64(waked)
            partial = waked & (crosswind > wake_radius - 1)
            relative_area = xr.where(
                partial,
                _lens_area(partial * crosswind, partial, partial * wake_radius)
                / np.pi,
                relative_area
            )
        else:
            relative_area = np.float64(
                is_downwind & (crosswind <= wake_radius))
        return (
            relative_area * induction_factor / np.square(wake_radius))

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

