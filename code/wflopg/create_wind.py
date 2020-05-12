import numpy as _np
import xarray as _xr

from wflopg.constants import COORDS
from wflopg.helpers import cyclic_extension as _cycext


def logarithmic_wind_shear(reference_height, roughness_length):
    """Return a logarithmic wind shear function"""
    def wind_shear(heights, speeds):
        """Return sheared wind speed at the given heights

        heights and speeds are assumed to be numpy arrays of compatible size.

        """
        if _np.all(heights == reference_height):
            return speeds
        else:
            return (speeds
                    * _np.log(heights / roughness_length)
                    / _np.log(reference_height / roughness_length))

    return wind_shear


def discretize_weibull(cweibull, cut_in, cut_out, speeds):
    """Return relevant speeds and wind speed probabilities"""
    # prepare the data structures used to discretize the Weibull
    # distribution
    wc = (speeds >= cut_in) & (speeds <= cut_out)  # within cut
    speeds = speeds[wc]
    speed_borders = _np.concatenate(([cut_in],
                                     (speeds[:-1] + speeds[1:]) / 2,
                                     [cut_out]))
    speed_bins = _xr.DataArray(
        _np.vstack((speed_borders[:-1], speed_borders[1:])).T,
        coords=[('speed', speeds), ('interval', COORDS['interval'])]
    )
    # Weibull CDF: 1 - exp(-(x/scale)**shape), so the probability for
    # an interval is
    # exp(-(xstart/scale)**shape) - exp(-(xend/scale)**shape)
    speed_bins = speed_bins / cweibull.sel(weibull_param='scale', drop=True)
    terms = _np.exp(
        - speed_bins ** cweibull.sel(weibull_param='shape', drop=True))
    speed_cpmf = (terms.sel(interval='lower', drop=True) -
                  terms.sel(interval='upper', drop=True))
    return speed_cpmf


def conformize_cpmf(speed_weights, cut_in, cut_out, speeds):
    """Return relevant speeds and wind speed probabilities"""
    speed_probs = speed_weights / speed_weights.sum(dim='speed')
    wc = (speeds >= cut_in) & (speeds <= cut_out)  # within cut
    speeds = speeds[wc]
    return speed_probs.sel(speed=speeds)


def subdivide(dir_weights, speed_probs, dir_subs,
              interpolation_method='linear'):
    """Return the subdivided wind direction and wind speed distributions

    The interpolation method can be 'nearest' or 'linear'. (Other options—such
    as 'cubic'—exist, but require even more careful handling of the cyclical
    nature of directions) The differences observed between the methods is small
    in our tests.

    """
    n = len(dir_weights.coords['direction'])
    dir_weights_cyc = _cycext(dir_weights, 'direction', 360)
    speed_probs_cyc = _cycext(speed_probs, 'direction', 360)
    dirs_cyc = dir_weights_cyc.coords['direction'].rename(direction='rel')
    # 'rel' is an ad hoc dimension for ‘local’ relative direction
    dirs_cyc.coords['rel'] = _np.linspace(0, 1, n + 1)
    dirs_interp = dirs_cyc.interp(
        rel=_np.linspace(0, 1, dir_subs * n, endpoint=False)).values

    #
    dir_weights = dir_weights_cyc.interp(direction=dirs_interp,
                                         method=interpolation_method)
    speed_probs = speed_probs_cyc.interp(direction=dirs_interp,
                                         method=interpolation_method)

    return dir_weights, speed_probs
