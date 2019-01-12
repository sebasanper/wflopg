import numpy as np
import xarray as xr
from ruamel.yaml import YAML as yaml

from wflopg import create_turbine
from wflopg import create_wind


class Owflop():
    """The main wind farm layout optimization problem object

    This object stores all the information and data related to a specific
    wind farm layout optimization problem.

    """
    def __init__(self):
        coords = {
            'xy_coord': ['x', 'y'],  # x ~ S→N, y ~ W→E
            'dc_coord': ['d', 'c']   # downwind/crosswind
        }
        # _ds is the main working Dataset
        self._ds = xr.Dataset(coords=coords)
        self._ds.xy_coord.attrs['description'] = (
            "Cartesian coordinates, where ‘x’ corresponds to the "
            "South-North direction and ‘y’ to the West-East direction.")
        self._ds.dc_coord.attrs['description'] = (
            "Cartesian coordinates determined by a given wind direction, "
            "where ‘d’ corresponds to the downwind direction "
            "and ‘c’ to the crosswind direction.")
        # history of layouts as a dict with identifier, layout pairs
        self.layouts = {}

    def load_problem(self, filename):
        """Load wind farm layout optimization problem file

        The file is assumed to be a YAML file in the format described by the
        schema https://bitbucket.org/equaeghe/pseudo_gradients-code/\
        src/master/schemata/wflo_problem-schema.yaml#.

        """
        with open(filename) as f:
            problem = yaml(typ='safe').load(f)
        # extract required parameters directly contained in problem document
        self.problem_uuid = problem['uuid']
        self.turbines = problem['turbines']
        # TODO: the following four parameters should be used to create the
        #       functions that calculate the powers and deficits
        self.objective = problem['objective']
        self.wake_model = problem['wake_model']
        self.wake_combination = problem['wake_combination']
        self.partial_wake = problem['partial_wake']
        # TODO: the following parameter should be used to, together with the
        #       rotor diameter, to generate the functions that check and
        #       correct the distance constraint
        self.turbine_distance = problem.get('turbine_distance', 0)

        # extract and process information and data from linked documents
        with open(problem['turbine']) as f:
            self.process_turbine(yaml(typ='safe').load(f))
        with open(problem['site']) as f:
            self.process_site(
                yaml(typ='safe').load(f),
                self.rotor_radius
            )
        with open(problem['wind_resource']) as f:
            self.process_wind_resource(
                yaml(typ='safe').load(f),
                self.roughness_length,
                problem.get('wind_direction_subdivisions', None),
                problem.get('wind_speeds', None),
                self.cut_in, self.cut_out
            )
        if 'layout' in problem:
            with open(problem['layout']) as f:
                initial_layout = yaml(typ='safe').load(f)['layout']
        else:
            initial_layout = [[0, 0]]
        self.process_layout(initial_layout)

    def process_turbine(self, turbine):
        self.rotor_radius = turbine['rotor_radius']
        self.hub_height = turbine['hub_height']
        self.rated_power = turbine['rated_power']
        self.rated_speed = turbine['rated_wind_speed']
        self.cut_in = turbine.get('cut_in', 0.0)
        self.cut_out = turbine.get('cut_out', np.inf)
        # define power curve
        if 'power_curve' in turbine:
            pc = np.array(turbine['power_curve'])
            self.power_curve = create_turbine.interpolated_power_curve(
                self.rated_power, self.rated_speed, self.cut_in, self.cut_out,
                pc
            )
        else:
            self.power_curve = create_turbine.cubic_power_curve(
                 self.rated_power, self.rated_speed, self.cut_in, self.cut_out)
        # define thrust curve
        if 'thrust_coefficient' in turbine:
            self.thrust_curve = create_turbine.constant_thrust_curve(
                                self.cut_in, self.cut_out, turbine['thrust_coefficient'])
        elif 'thrust_curve' in turbine:
            tc = np.array(turbine['thrust_curve'])
            self.thrust_curve = create_turbine.interpolated_thrust_curve(
                                                 self.cut_in, self.cut_out, tc)
        else:
            raise ValueError("Turbine document should contain either "
                             "a 'thrust_curve' or "
                             "a constant 'thrust_coefficient'")

    def process_site(self, site, rotor_radius):
        self.rotor_constraints = site['rotor_constraints']
        self.site_radius = site['radius']
        # TODO: import site parcels and boundaries and together with the rotor
        #       radius create functions to visualize and check the boundary
        #       constraints
        self.roughness_length = site.get('roughness', None)

    def process_wind_resource(self, wind_resource, roughness_length,
                              dir_subs, speeds, cut_in, cut_out):
        self.reference_height = wind_resource['reference_height']
        self.air_density = wind_resource.get('air_density', 1.2041)
        self.atmospheric_stability = wind_resource.get(
            'atmospheric_stability', None)
        self.turbulence_intensity = wind_resource.get(
            'turbulence_intensity', None)

        # Create the wind shear function
        self.wind_shear = create_wind.logarithmic_wind_shear(
                                       self.reference_height, roughness_length)

        wind_rose = wind_resource['wind_rose']

        # Create wind direction probability mass function
        dirs = np.array(wind_rose['directions'])
        dir_weights = np.array(wind_rose['direction_pmf'])
        # we do not assume the wind directions are sorted in the data file and
        # therefore sort them here
        dir_sort_index = dirs.argsort()
        dirs = dirs[dir_sort_index]
        dir_weights = dir_weights[dir_sort_index]

        # Create the conditional wind speed probability mass function
        #
        # NOTE: All of this is at reference height; don't forget to apply wind
        #       shear on use!
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
            speeds = np.array(speeds)
            speeds = speeds[(speeds >= cut_in) & (speeds <= cut_out)]
            speed_borders = np.concatenate(([cut_in],
                                            (speeds[:-1] + speeds[1:]) / 2,
                                            [cut_out]))
            speed_bins = xr.DataArray(
                np.vstack((speed_borders[:-1], speed_borders[1:])).T,
                coords=[('wind_speed', speeds), ('bound', ['start', 'end'])]
            )
            cweibull = xr.DataArray(
                wind_rose['speed_cweibull'],
                coords=[('direction', dirs), ('param', ['scale', 'shape'])]
            )
            # Weibull CDF: 1 - exp(-(x/scale)**shape), so the probability for
            # an interval is
            # exp(-(xstart/scale)**shape) - exp(-(xend/scale)**shape)
            speed_bins = speed_bins / cweibull.loc[:, 'scale']
            terms = np.exp(- speed_bins ** cweibull.loc[:, 'shape'])
            speed_cpmf = terms.sel(bound='start') - terms.sel(bound='end')
            speed_weights = speed_cpmf.values.T
        elif 'speed_cpmf' in wind_rose and 'speeds' in wind_rose:
            speeds = np.array(wind_rose['speeds'])
            speed_weights = np.array(wind_rose['speed_cpmf'])
            wc = (speeds >= cut_in) & (speeds <= cut_out)  # within cut
            speeds = speeds[wc]
            speed_weights = speed_weights[:, wc]
        else:
            raise ValueError(
                "A conditional wind speed probability distribution "
                "should be given either as parameters for conditional Weibull "
                "distributions or as a conditional probability mass function.")

        # Subdivide wind direction and speed pmfs if needed
        if dir_subs:
            dirs_cyc = np.concatenate((dirs, 360 + dirs[:1]))
            dir_weights_cyc = xr.DataArray(
                np.concatenate((dir_weights, dir_weights[:1])),
                coords=[('direction', dirs_cyc)]
            )
            speed_weights_cyc = xr.DataArray(
                np.concatenate((speed_weights, speed_weights[:1])),
                coords=[('direction', dirs_cyc), ('wind_speed', speeds)]
            )
            dirs_cyc = xr.DataArray(
                dirs_cyc,
                coords=[('rel', np.linspace(0., 1., len(dirs) + 1))]
            )
            dirs_interp = dirs_cyc.interp(
                rel=np.linspace(0., 1., dir_subs * len(dirs) + 1)
            ).values
            dirs_interp = dirs_interp[:-1]  # drop the last, cyclical value

            # the interpolation method can be 'nearest' or 'linear'
            # (other options—such as 'cubic'—exist, but require even more
            # careful handling of the cyclical nature of directions)
            dir_weights = dir_weights_cyc.interp(direction=dirs_interp,
                                                 method='linear')
            speed_weights = speed_weights_cyc.interp(direction=dirs_interp,
                                                     method='linear')
        else:
            dir_weights = xr.DataArray(dir_weights,
                                       coords=[('direction', dirs)])
            speed_weights = xr.DataArray(
                speed_weights,
                coords=[('direction', dirs), ('wind_speed', speeds)]
            )

        # Store pmfs; obtain them from the weight arrays by normalization
        self._ds['direction_pmf'] = dir_weights / dir_weights.sum()
        self._ds['wind_speed_cpmf'] = (speed_weights /
                                       speed_weights.sum(dim='wind_speed'))

        # Store downwind unit vectors
        # Convert inflow wind direction
        # - from windrose (N=0, CW) to standard (E=0, CCW): 90 - wind_dir
        # - from upwind to downwind: +180
        # - from degrees to radians
        directions_rad = np.radians(90 - self._ds.coords['direction'] + 180)
        self._ds['downwind'] = xr.DataArray(
            np.array([np.cos(directions_rad), np.sin(directions_rad)]).T,
            dims=['direction', 'xy_coord']
        )

    def process_layout(self, initial_layout):
        # turbines affected by the wake
        self._ds['layout'] = xr.DataArray(initial_layout,
                                          dims=['target', 'xy_coord'])
        self.layouts['initial'] = xr.DataArray(
            initial_layout,
            dims=['target', 'xy_coord'],
            coords={'xy_coord': self._ds.coords['xy_coord']}
        )
        # turbines causing the wakes
        # NOTE: currently, these are the same as the ones affected
        self._ds['context'] = xr.DataArray(initial_layout,
                                           dims=['source', 'xy_coord'])

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
