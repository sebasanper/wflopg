import numpy as np
import xarray as xr

from wflopg import COORDS


def logarithmic_wind_shear(reference_height, roughness_length):
    """Return a logarithmic wind shear function"""
    def wind_shear(heights, speeds):
        """Return sheared wind speed at the given heights

        heights and speeds are assumed to be numpy arrays of compatible size.

        """
        if np.all(heights == reference_height):
            return speeds
        else:
            return (speeds * np.log(heights / roughness_length)
                           / np.log(reference_height / roughness_length))

    return wind_shear


def sort_directions(dirs, dir_weights):
    """Return sorted wind direction and mass arrays"""
    dirs = np.array(dirs)
    dir_weights = np.array(dir_weights)
    # we do not assume the wind directions are sorted in the data file and
    # therefore sort them here
    dir_sort_index = dirs.argsort()
    dirs = dirs[dir_sort_index]
    dir_weights = dir_weights[dir_sort_index]
    return dirs, dir_weights


def discretize_weibull(weibull, speeds, cut_in, cut_out):
    """Return relevant speeds and wind speed probabilities"""
    # prepare the data structures used to discretize the Weibull
    # distribution
    speeds = speeds[(speeds >= cut_in) & (speeds <= cut_out)]
    speed_borders = np.concatenate(([cut_in],
                                    (speeds[:-1] + speeds[1:]) / 2,
                                    [cut_out]))
    speed_bins = xr.DataArray(
        np.vstack((speed_borders[:-1], speed_borders[1:])).T,
        coords=[('speed', speeds), ('interval', COORDS['interval'])]
    )
    cweibull = xr.DataArray(
        weibull,
        dims=['direction', 'weibull_param'],
        coords={'weibull_param': COORDS['weibull_param']}
    )
    # Weibull CDF: 1 - exp(-(x/scale)**shape), so the probability for
    # an interval is
    # exp(-(xstart/scale)**shape) - exp(-(xend/scale)**shape)
    speed_bins = speed_bins / cweibull.sel(weibull_param='scale', drop=True)
    terms = np.exp(
        - speed_bins ** cweibull.sel(weibull_param='shape', drop=True))
    speed_cpmf = (terms.sel(interval='lower', drop=True) -
                  terms.sel(interval='upper', drop=True))
    return speeds, speed_cpmf.values.T


def conformize_cpmf(speed_weights, speeds, cut_in, cut_out):
    """Return relevant speeds and wind speed probabilities"""
    speed_weights = xr.DataArray(
        speed_weights, dims=['direction', 'speed'])
    speed_probs = speed_weights / speed_weights.sum(dim='speed')
    wc = (speeds >= cut_in) & (speeds <= cut_out)  # within cut
    speeds = speeds[wc]
    return speeds, speed_probs.sel(speed=wc).values


def subdivide(dirs, speeds, dir_weights, speed_probs, dir_subs,
              interpolation_method='linear'):
    """Return the subdivided wind direction and wind speed distributions

    The interpolation method can be 'nearest' or 'linear'. (Other options—such
    as 'cubic'—exist, but require even more careful handling of the cyclical
    nature of directions) The differences observed between the methods is small
    in our tests.

    """
    dirs_cyc = np.concatenate((dirs, 360 + dirs[:1]))
    dir_weights_cyc = xr.DataArray(
        np.concatenate((dir_weights, dir_weights[:1])),
        coords=[('direction', dirs_cyc)]
    )
    speed_probs_cyc = xr.DataArray(
        np.concatenate((speed_probs, speed_probs[:1])),
        coords=[('direction', dirs_cyc), ('speed', speeds)]
    )
    dirs_cyc = xr.DataArray(
        dirs_cyc,
        coords=[('rel', np.linspace(0., 1., len(dirs) + 1))]
        # 'rel' is an ad-hoc dimension for ‘local’ relative direction
    )
    dirs_interp = dirs_cyc.interp(
        rel=np.linspace(0., 1., dir_subs * len(dirs) + 1)
    ).values
    dirs_interp = dirs_interp[:-1]  # drop the last, cyclical value

    #
    dir_weights = dir_weights_cyc.interp(direction=dirs_interp,
                                         method=interpolation_method)
    speed_probs = speed_probs_cyc.interp(direction=dirs_interp,
                                         method=interpolation_method)

    return dir_weights, speed_probs
