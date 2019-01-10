import numpy as np
import xarray as xr
from ruamel.yaml import YAML as yaml

import .create_turbine
import .create_wind


class Owflop():
    """The main wind farm layout optimization problem object

    This object stores all the information and data related to a specific
    wind farm layout optimization problem.

    """
    def __init__(self):
        coords = [
            ('xy_coord', ['x', 'y']),  # x ~ S→N, y ~ W→E
            ('dc_coord', ['d', 'c'])   # downwind/crosswind
        ]
        # _ds is the main working Dataset
        self._ds = xr.Dataset(coords=coords)
        self._ds.xy_coord.attrs['description'] = (
            "Cartesian coordinates, where ‘x’ corresponds to the "
            "South-North direction and ‘y’ to the West-East direction.")
        self._ds.dc_coord.attrs['description'] = (
            "Cartesian coordinates determined by a given wind direction, "
            "where ‘d’ corresponds to the downwind direction "
            "and ‘y’ to the crosswind direction.")
        # _history is the Dataset containing a history of layouts
        self._history = xr.Dataset(coords=[coords[0]])
        self._history.xy_coord.attrs['description'] = (
                                        self._ds.xy_coord.attrs['description'])
        # A single turbine at the origin as default initial layout
        initial_layout = [[0, 0]]
        self._ds['layout'] = xr.DataArray(
            np.array(initial_layout, np.float32),
            dims=['target', 'xy_coord']
        )
        self._history['initial'] = xr.DataArray(
            np.array(initial_layout, np.float32),
            dims=['target_initial', 'xy_coord'],
            attrs={'description': 'initial layout'}
        )

    def load_problem(self, filename):
        """Load wind farm layout optimization problem file

        The file is assumed to be a YAML file in the format described by the
        schema https://bitbucket.org/equaeghe/pseudo_gradients-code/\
        src/master/schemata/wflo_problem-schema.yaml#.

        """
        problem = yaml(typ='safe').load(filename)
        # extract required parameters directly contained in problem document
        self.problem_uuid = problem['uuid']
        self.turbines = problem['turbines']
        # TODO: objective?
        self.wake_model = problem['wake_model']
        self.wake_combination = problem['wake_combination']
        self.partial_wake = problem['partial_wake']
        self.turbine_distance = problem.get('turbine_distance', 0)

        # extract and process information and data from linked documents
        cut_in, cut_out = self.process_turbine(
                                     yaml(typ='safe').load(problem['turbine']))
        roughness_length = self.process_site(
                                        yaml(typ='safe').load(problem['site']))
        self.process_wind_resource(
            yaml(typ='safe').load(problem['wind_resource']),
            roughness_length,
            problem.get('wind_direction_subdivisions', None),
            problem.get('wind_speeds', None),
            cut_in, cut_out
        )
        if 'layout' in problem:
            self.process_layout(
                            yaml(typ='safe').load(problem['layout'])['layout'])


    def process_turbine(turbine):
        self.rotor_radius = turbine['rotor_radius']
        self.hub_height = turbine['hub_height']
        rated_power = turbine['rated_power']
        rated_speed = turbine['rated_wind_speed']
        cut_in = turbine.get('cut_in', 0.0)
        cut_out = turbine.get('cut_out', np.inf)
        # define power curve
        if 'power_curve' in turbine:
            pc = np.array(turbine['power_curve'], np.float32)
            self.power_curve = create_turbine.interpolated_power_curve(
                                 rated_power, rated_speed, cut_in, cut_out, pc)
        else:
            self.power_curve = create_turbine.cubic_power_curve(
                                     rated_power, rated_speed, cut_in, cut_out)
        # define thrust curve
        if 'thrust_coefficient' in turbine:
            self.thrust_curve = create_turbine.constant_thrust_curve(
                                cut_in, cut_out, turbine['thrust_coefficient'])
        elif 'thrust_curve' in turbine:
            tc = np.array(turbine['thrust_curve'], np.float32)
            self.thrust_curve = create_turbine.interpolated_thrust_curve(
                                                           cut_in, cut_out, tc)
        else:
            raise ValueError("Turbine document should contain either "
                             "a 'thrust_curve' or "
                             "a constant 'thrust_coefficient'")

        return cut_in, cut_out


    def process_site(site):
        self.rotor_constraints = site['rotor_constraints']
        self.site_radius = site['radius']
        # TODO: import site parcels and boundaries
        return site.get('roughness', None)


    def process_wind_resource(wind_resource, roughness_length,
                              dir_subs, speeds, cut_in, cut_out):
        reference_height = wind_resource['reference_height']
        self.air_density = wind_resource.get('air_density', 1.2041)
        self.atmospheric_stability = wind_resource.get(
            'atmospheric_stability', None)
        self.turbulence_intensity = wind_resource.get(
            'turbulence_intensity', None)

        # Create the wind shear function
        self.wind_shear = create_wind.logarithmic_wind_shear(reference_height,
                                                             roughness_length)

        # Create and store the wind direction distribution
        wind_rose = wind_resource['wind_rose']
        dir_pmf = np.array([wind_rose['directions'],
                            wind_rose['direction_pmf']], np.float32).T
        dir_pmf = dir_pmf[dirs.argsort()]
        dirs = dir_pmf[:, 0]
        # optionally subdivide windrose
        if dir_subs:
            dir_pmf = np.repeat(dir_pmf, dir_subs, axis=0)
            dirs_cy = np.concatenate(([dirs[-1] - 360], dirs, [dirs[0] + 360]))
            dir_bins = np.array([(dirs_cy[1:-1] + dirs_cy[:-2]) / 2,
                                 (dirs_cy[2:] + dirs_cy[1:-1]) / 2]).T
            for k, bounds in enumerate(dir_bins):
                dir_pmf[k*dir_subs:(k+1)*dir_subs] = np.linspace(
                                bounds[0], bounds[1], 2 * dir_subs + 1)[1:-1:2]
        # normalize wind rose (before we just had weights, not probabilities)
        dir_pmf[:, 1] /= np.sum(dir_pmf[:, 1])
        # add direction probability mass function to the object's Dataset
        self._ds['direction_pmf'] = xr.DataArray(
                      dir_pmf[:, 1], dims=['directions'], coords=dir_pmf[:, 0])

        # Create and store the conditional wind speed probability mass function
        #
        # NOTE: All at reference height; don't forget to apply wind shear on
        #       use!
        #
        # NOTE: Of the conditional wind speed probability mass function, only
        #       the values within the [cut_in, cut_out] interval are stored in
        #       self._ds['wind_speed_cpmf'] (defined below), as the others give
        #       no contribution to the power production. (This part may need to
        #       be revised if, e.g., we want to take into account loads, where
        #       wind speeds above cut_out are certainly relevant.)
        #
        if 'speed_cweibull' in wind_rose:
            if not speeds:
                raise ValueError(
                    "An array of wind speeds must be specified in case the "
                    "wind resource is formulated in terms of Weibull "
                    "distributions")
            # prepare the data structures used to discretize the Weibull
            # distribution
            speeds = speeds[speeds >= cut_in & speeds <= cut_out]
            speed_borders = np.concatenate(([cut_in],
                                            (speeds[:-1] + speeds[1:]) /2,
                                            [cut_out]))
            speed_bins = xr.DataArray(
                [speed_borders[:-1], speed_borders[1:]]).T,
                coords=[('wind_speed', speeds), ('bound', ['start', 'end'])]
            )
            cweibull = xr.DataArray(
                wind_rose['speed_cweibull'],
                coords=[('directions', dir_pmf[:, 0]),
                        ('param', ['scale', 'shape'])]
            )
            # Weibull CDF: 1 - exp(-(x/scale)**shape), so the probability for
            # an interval is
            # exp(-(xstart/scale)**shape) - exp(-(xend/scale)**shape)
            speed_bins /= cweibull.loc[:, 'scale']
            terms = np.exp(- speed_bins ** cweibull.loc[:, 'shape'])
            self._ds['wind_speed_cpmf'] = (terms.sel(bound='start') -
                                           terms.sel(bound='end'))
        elif 'speed_cpmf' in wind_rose and 'speeds' in wind_rose:
            speeds = np.array(wind_rose['speeds'], np.float32)
            speed_cpmf = np.array(wind_rose['speed_cpmf'], np.float32)
            speed_cpmf /= np.sum(speed_cpmf, axis=1)  # normalize weights
            wc = speeds >= cut_in & speeds <= cut_out # within cut
            self._ds['wind_speed_cpmf'] = xr.DataArray(
                speed_cpmf[:, wc]),
                coords=[('directions', dir_pmf[:, 0]), ('wind_speed', speeds)]
            )
        else:
          raise ValueError(
            "A conditional wind speed probability distribution "
            "should be given either as parameters for conditional Weibull "
            "distributions or as a conditional probability mass function.")


    def process_layout(initial_layout):
        self._ds['layout'].values = np.array(initial_layout, np.float32)
        self._history['layout'].values = np.array(initial_layout, np.float32)


# dimensions = {'source', 'target',  # source and target turbine dummy dimensions
#               'xy_coord', 'dc_coord',
#               'direction', 'wind_speed'} # we assume a rectangular grid
#
# variables = {
#     'context': ['source', 'xy_coord'],  # turbines causing wakes (includes 'target')
#     'layout': ['target', 'xy_coord'],  # turbines whose positions we can affect
#     'distance': ['source', 'turbine'],  # distances between source and target turbines
#     'downwind': ['direction', 'xy_coord'],  # downwind unit vectors
#     'vector': ['source', 'target', 'direction', 'dc_coord'],  # downwind/crosswind coordinates for vectors between all source and target turbines, for all directions
#     'deficit': ['source', 'target', 'direction', 'wind_speed'],  # speed deficit
#     'combined_deficit': ['target', 'direction', 'wind_speed'],
#     'relative_deficit': ['source', 'target', 'direction', 'wind_speed'],
#     'power': ['target', 'direction', 'wind_speed'],
#     'loss': ['target', 'direction', 'wind_speed'],
#     'blamed_loss': ['source', 'target', 'direction', 'wind_speed'],
#     'blamed_loss_vector': ['source', 'target', 'direction', 'wind_speed', 'xy_coord'],
#     'wind_speed_pmf': ['direction', 'wind_speed'],  # 'wind_speed' pmf conditional on 'direction'
#     'direction_pmf': ['direction'],
#     'expected_power': ['target']
# }
