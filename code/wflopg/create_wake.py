import numpy as _np
import xarray as _xr


def _common(dc_vector_adim):
    downwind = dc_vector_adim.sel(dc='d', drop=True)
    crosswind = _np.abs(dc_vector_adim.sel(dc='c', drop=True))
    is_downwind = downwind > 0
    return downwind, crosswind, is_downwind


def _lens_area(d, r, R):
    """Return relative area of half-lens

    This function is used to calculate the area of a lens with radii r and R at
    distance d.

    From <http://mathworld.wolfram.com/Lens.html>, modified for efficiency.

    (Alternative: <https://christianhill.co.uk/blog/overlapping-circles/>)

    """
    dd = _np.square(d)
    rr = _np.square(r)
    RR = _np.square(R)
    RR_rr = RR - rr
    return (
        rr * _np.arccos((dd - RR_rr) / (2 * d * r))
        +
        RR * _np.arccos((dd + RR_rr) / (2 * d * R))
        -
        _np.sqrt(2 * dd * (RR + rr) - _np.square(dd) - _np.square(RR_rr)) / 2
    )


def _relative_area_function(averaging):
    """Return the function that computes the relative waked area"""
    def hub_waked(is_downwind, crosswind, wake_radius):
        """Calculate whether the hub is waked

        This will result in no rotor plane averaging to be done.

        """
        return (is_downwind & (crosswind <= wake_radius)).astype(_np.float64)

    def relative_waked_area(is_downwind, crosswind, wake_radius):
        """Calculate the relative waked rotor area

        This will result in rotor plane averaging to be done
        to take into account partial waking.

        """
        rel_crosswind = crosswind - wake_radius
        waked = is_downwind & (rel_crosswind < 1)
        partial = waked & (rel_crosswind > - 1)
        return _xr.where(
            partial,
            _lens_area(partial * crosswind, partial, partial * wake_radius)
            / _np.pi,
            waked.astype(_np.float64)
        )

    return relative_waked_area if averaging else hub_waked


def linear_top_hat(thrust_curve, rotor_radius, expansion_coeff=None,
                   deficit_type='Jensen', stream_tube_assumption='rotor',
                   averaging=False):
    """Return a linearly expanding top hat wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    """
    induction_factor = 1 - _np.sqrt(1 - thrust_curve)

    # Deal with stream tube assumption
    if stream_tube_assumption == 'rotor':
        stream_tube_radius = 1
    elif stream_tube_assumption == 'Frandsen':
        # Use (adim.) stream tube radius downwind of the rotor
        # instead of the rotor radius itself.
        # [Frandsen, S. (1992) On the wind speed reduction in the center of
        #  large clusters of wind turbines. Journal of Wind Engineering and
        #  Industrial Aerodynamics 39:251–265. Section 2.2]
        stream_tube_radius = _np.sqrt(
            # TODO: deal with case induction_factor == 1
            (1 - 0.5 * induction_factor) / (1 - induction_factor))
    else:
        raise ValueError("Not a valid stream tube assumption: "
                         " ‘{}’.".format(stream_tube_assumption))

    # Deal with deficit type
    if deficit_type == 'Jensen':
        def deficit(inv_rel_wake_area):
            """Return Jensen wake model deficit"""
            return induction_factor * inv_rel_wake_area
    elif deficit_type == 'Frandsen':
        def deficit(inv_rel_wake_area):
            """Return Frandsen et al. wake model deficit

            This implements doi:10.1002/we.189 (11).
            (But still assumes linear expansion, i.e., k=1 for (13)!)

            """
            return (
                0.5 * (1 - _np.sqrt(1 - 2 * thrust_curve * inv_rel_wake_area)))
    else:
        raise ValueError(
            "Not a valid deficit type: ‘{}’.".format(deficit_type))

    relative_area = _relative_area_function(averaging)

    def wake_model(dc_vector):
        """Return wind speed deficit due to wake

        The argument must be an xarray DataArray of dimensional
        downwind-crosswind coordinate pairs.

        """
        downwind, crosswind, is_downwind = _common(dc_vector / rotor_radius)
        wake_radius = stream_tube_radius + expansion_coeff * downwind
        del downwind
        rel_waked_area = relative_area(is_downwind, crosswind, wake_radius)
        del is_downwind, crosswind
        inv_rel_wake_area = _np.square(stream_tube_radius / wake_radius)
        del wake_radius
        return rel_waked_area * deficit(inv_rel_wake_area)

    return wake_model


def entrainment(thrust_curve, rotor_radius, entrainment_coeff=0.15,
                averaging=False):
    """Return an entrainment wake model function

    The thrust_curve must be an xarray DataArray with as a single dimension the
    wind speed, whose coordinate values must be those free stream wind speeds
    for which the wake deficit must be calculated. The other arguments are
    scalar values for the quantities described by their name.

    This implements doi:10.1088/1742-6596/1037/7/072019 (25) with x_i=0.

    """
    # TODO: deal with case thrust_curve == 0
    offset = (
        (1 - thrust_curve) ** .75 / (1 - _np.sqrt(1 - thrust_curve)) ** 1.5)
    scaler = _np.sqrt(0.5 * thrust_curve)

    relative_area = _relative_area_function(averaging)

    def wake_model(dc_vector):
        # TODO: verify that the code produces the right geometry & deficit!
        downwind, crosswind, is_downwind = _common(dc_vector / rotor_radius)
        downwind_factor = _np.cbrt(
            6 * entrainment_coeff / scaler * downwind + offset)
        del downwind
        wake_radius = scaler * (downwind_factor + 1 / downwind_factor)
        rel_waked_area = relative_area(is_downwind, crosswind, wake_radius)
        del crosswind, is_downwind, wake_radius
        return rel_waked_area / (1 + _np.square(downwind_factor))

    return wake_model


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
    sigma_at_source = 1 / _np.sqrt(2)

    def wake_model(dc_vector):
        """Return wind speed deficit due to wake

        The argument must be an xarray DataArray of dimensional
        downwind-crosswind coordinate pairs.

        """
        downwind, crosswind, is_downwind = _common(dc_vector / rotor_radius)
        # multiplication with is_downwind to avoid negative radical later
        sigma = sigma_at_source + expansion_coeff * downwind * is_downwind
        exponent = -(crosswind / sigma) ** 2 / 2
        radical = 1 - thrust_curve / (2 * sigma ** 2)
        return is_downwind * (1. - _np.sqrt(radical)) * _np.exp(exponent)

    return wake_model


def rss_combination():
    """Return the root-sum-square wake deficit combination rule"""
    def combination_rule(deficit):
        """Return combined and relative wake deficits

        deficit must be an xarray DataArray of individual deficits with
        'source' as one dimension.

        """
        squared = _np.square(deficit)
        squared_combined = squared.sum(dim='source')
        # we add 1 where the denominator (and numerator) would be 0
        relative = squared / (squared_combined + (squared_combined == 0))
        # RSS does not guarantee <= 1
        squared_combined_saturated = _np.minimum(squared_combined, 1)
        return _np.sqrt(squared_combined_saturated), relative

    return combination_rule


def no_combination():
    """Return the combination rule for cases where combination is unneeded"""
    def combination_rule(deficit):
        """Return combined and relative wake deficits

        deficit must be an xarray DataArray of individual deficits with
        'source' as one dimension, where its length is 1.

        """
        squeezed = deficit.squeeze('source')
        return squeezed, _xr.ones_like(squeezed)

    return combination_rule
