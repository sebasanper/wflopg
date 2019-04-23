import numpy as _np
import xarray as _xr
from ruamel.yaml import YAML as yaml

from wflopg.constants import COORDS
from wflopg import create_turbine
from wflopg import create_site
from wflopg import create_wind
from wflopg import create_wake
from wflopg import layout_geometry
from wflopg import create_constraint


class Owflop():
    """The main wind farm layout optimization problem object

    This object stores all the information and data related to a specific
    wind farm layout optimization problem.

    """
    def __init__(self):
        # _ds is the main working Dataset
        self._ds = _xr.Dataset(coords={dim: COORDS[dim]
                                      for dim in {'xy', 'dc'}})
        # history of layouts and friends as a list of _xr.DataSets
        self.history = []

    def load_problem(self, filename,
                     wind_resource_filename=None,
                     layout_filename=None,
                     hex_layout=False, hex_proximity_factor=None):
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

        # extract and process information and data from linked documents
        with open(problem['turbine']) as f:
            self.process_turbine(yaml(typ='safe').load(f))
        with open(problem['site']) as f:
            self.process_site(yaml(typ='safe').load(f))
        if wind_resource_filename is None:
            wind_resource_filename = problem['wind_resource']
        with open(wind_resource_filename) as f:
            self.process_wind_resource(
                yaml(typ='safe').load(f),
                self.roughness_length,
                problem.get('wind_direction_subdivisions', None),
                problem.get('wind_speeds', None),
                self.cut_in, self.cut_out
            )

        # calculate power information for which no wake calculations are needed
        self.calculate_wakeless_power()

        # process information for wake model-related properties
        self.process_wake_model(
            problem['wake_model'],
            problem.get('expansion_coefficient', None),
            problem['wake_combination']
        )
        self.process_objective(problem['objective'])

        # Store downwind and crosswind unit vectors
        self._ds['downwind'] = layout_geometry.generate_downwind(
            self._ds.direction)
        self._ds['crosswind'] = layout_geometry.generate_crosswind(
            self._ds.downwind)

        # deal with initial layout
        if layout_filename is not None:
            with open(layout_filename) as f:
                initial_layout = yaml(typ='safe').load(f)['layout']
        elif ('layout' in problem) and not hex_layout:
            with open(problem['layout']) as f:
                initial_layout = yaml(typ='safe').load(f)['layout']
        else:
            # (random) number of turbines
            if hex_layout is True:
                turbines = self.turbines
            elif isinstance(hex_layout, int):
                turbines = hex_layout
            elif isinstance(self.turbines, list):
                turbines = _np.random.randint(*self.turbines)
            initial_layout = self.create_hex_layout(turbines)
        self.process_initial_layout(initial_layout)
        self.calculate_geometry()

        # create function to generate turbine constraint violation fixup steps
        if hex_layout and hex_proximity_factor:
            self.minimal_proximity = (self._ds.layout.hex_distance
                                      * hex_proximity_factor)
        else:
            self.minimal_proximity = (
                    problem.get('turbine_distance', 1)
                    * (2 * self.rotor_radius) / self.site_radius)
        self.proximity_repulsion = (
            create_constraint.distance(self.minimal_proximity))

    def create_hex_layout(self, turbines):
        # create squarish hexagonal—so densest—packing to cover site
        max_turbines = 0
        factor = 1
        while max_turbines < turbines:
            x_step = _np.sqrt(factor / turbines) * 2
            y_step = x_step * _np.sqrt(3) / 2
            n = int(1 / x_step) + 1
            m = int(1 / y_step)
            xs = _np.arange(-n, n) * x_step
            ys = _np.arange(-m, m + 1) * y_step
            mg = _np.meshgrid(xs, ys)
            mg[0] = (mg[0].T + (_np.arange(-m, m+1) % 2) * x_step / 2).T
            covering_layout = _xr.DataArray(
                _np.stack([mg[0].ravel(), mg[1].ravel()], axis=-1),
                dims=['target', 'uv'], coords={'uv': ['u', 'v']}
            )
            # add random offset
            offset = _xr.DataArray(
                _np.random.random(2) * _np.array([x_step, y_step]),
                coords=[('uv', ['u', 'v'])]
            )
            covering_layout += offset
            # rotate over random angle
            angle = _np.random.random() * _np.pi / 3  # hexgrid is π/3-symmetric
            cos_angle = _np.cos(angle)
            sin_angle = _np.sin(angle)
            rotation_matrix = _xr.DataArray(
                _np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]),
                coords=[('uv', ['u', 'v']), ('xy', COORDS['xy'])]
            )
            rotated_covering_layout = covering_layout.dot(rotation_matrix)
            # only keep turbines inside
            inside = self.inside(rotated_covering_layout)
            dense_layout = rotated_covering_layout[
                inside['in_site'].rename(position='target')]
            dense_layout.attrs['hex_distance'] = x_step
            # TODO: avoid the rename in some way
            max_turbines = len(dense_layout)
            factor *= max_turbines / turbines
        return dense_layout[
            _np.random.choice(max_turbines, turbines, replace=False)]

    def process_turbine(self, turbine):
        self.rotor_radius = turbine['rotor_radius']
        self.hub_height = turbine['hub_height']
        self.rated_power = turbine['rated_power']
        self.rated_speed = turbine['rated_wind_speed']
        self.cut_in = turbine.get('cut_in', 0.0)
        self.cut_out = turbine.get('cut_out', _np.inf)
        # define power curve
        if 'power_curve' in turbine:
            pc = _np.array(turbine['power_curve'])
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
            tc = _np.array(turbine['thrust_curve'])
            self.thrust_curve = create_turbine.interpolated_thrust_curve(
                                                 self.cut_in, self.cut_out, tc)
        else:
            raise ValueError("Turbine document should contain either "
                             "a 'thrust_curve' or "
                             "a constant 'thrust_coefficient'")

    def process_site(self, site):
        self.roughness_length = site.get('roughness', None)
        self.site_radius = site['radius'] * 1e3  # km to m
        # Process parcels
        if 'parcels' in site:
            self.parcels = create_site.parcels(
                site['parcels'], self.rotor_radius / self.site_radius)
        # create function that reports whether a turbine is inside the site
        self.inside = create_constraint.inside_site(self.parcels)
        # create function to generate site constraint violation fixup steps
        self.to_border = create_constraint.site(self.parcels)
        # Process boundaries
        if 'boundaries' in site:
            self.boundaries = create_site.boundaries(site['boundaries'])

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

        # Sort wind direction and mass function
        dirs, dir_weights = create_wind.sort_directions(
            wind_rose['directions'], wind_rose['direction_pmf'])

        # Create the conditional wind speed probability mass function
        #
        # NOTE: Of the conditional wind speed probability mass function, only
        #       the values within the [cut_in, cut_out] interval are stored in
        #       self._ds['wind_speed_cpmf'] (defined below), as the others give
        #       no contribution to the power production.
        #
        if 'speed_cweibull' in wind_rose:
            if not speeds:
                raise ValueError(
                    "An array of wind speeds must be specified in case the "
                    "wind resource is formulated in terms of Weibull "
                    "distributions")
            # take wind shear into account
            speeds = self.wind_shear(self.hub_height, _np.array(speeds))
            speeds, speed_probs = create_wind.discretize_weibull(
                wind_rose['speed_cweibull'], speeds, cut_in, cut_out)
        elif 'speed_cpmf' in wind_rose and 'speeds' in wind_rose:
            # take wind shear into account
            speeds = self.wind_shear(self.hub_height,
                                     _np.array(wind_rose['speeds']))
            speeds, speed_probs = create_wind.conformize_cpmf(
                wind_rose['speed_cpmf'], speeds, cut_in, cut_out)
        else:
            raise ValueError(
                "A conditional wind speed probability distribution "
                "should be given either as parameters for conditional Weibull "
                "distributions or as a conditional probability mass function.")

        # Subdivide wind direction and speed pmfs if needed
        if dir_subs:
            dir_weights, speed_probs = create_wind.subdivide(
                dirs, speeds, dir_weights, speed_probs, dir_subs)
        else:
            dir_weights = _xr.DataArray(dir_weights,
                                       coords=[('direction', dirs)])
            speed_probs = _xr.DataArray(
                speed_probs,
                coords=[('direction', dirs), ('speed', speeds)]
            )

        # normalize direction pmf
        dir_probs = dir_weights / dir_weights.sum()

        # Store pmfs; obtain them from the weight arrays by normalization
        self._ds['direction_pmf'] = dir_probs
        self._ds['wind_speed_cpmf'] = speed_probs

    def process_initial_layout(self, initial_layout):
        # turbines affected by the wake
        self._ds['layout'] = _xr.DataArray(
            initial_layout,
            dims=['target', 'xy'],
            coords={'target': range(len(initial_layout))}
        )
        # turbines causing the wakes
        # NOTE: currently, these are the same as the ones affected
        self._ds['context'] = self._ds.layout.rename(target='source')

    def process_wake_model(self,
                           model, expansion_coefficient, combination_rule):
        thrusts = self.thrust_curve(self._ds.speed)
        # preliminaries for wake model definition
        if model.startswith("Jensen"):
            if not expansion_coefficient:
                expansion_coefficient = (
                    0.5 / _np.log(self.hub_height / self.roughness_length))
        # define wake model
        if model == "BPA (IEA37)":
            self.wake_model = create_wake.bpa_iea37(
                thrusts,
                self.rotor_radius,
                self.turbulence_intensity
            )
        elif model == "Jensen":
            self.wake_model = create_wake.jensen(
                thrusts, self.rotor_radius, expansion_coefficient)
        elif model == "Jensen according to Frandsen":
            self.wake_model = create_wake.jensen_frandsen(
                thrusts, self.rotor_radius, expansion_coefficient)
        elif model == "Jensen with partial wake":
            self.wake_model = create_wake.jensen_averaged(
                thrusts, self.rotor_radius, expansion_coefficient)
        elif model == "Jensen according to Frandsen with partial wake":
            self.wake_model = create_wake.jensen_frandsen_averaged(
                thrusts, self.rotor_radius, expansion_coefficient)
        else:
            raise ValueError("Unkown wake model specified.")
        # define combination rule
        if combination_rule == "RSS":
            self.combination_rule = create_wake.rss_combination()
        else:
            raise ValueError("Unknown wake combination rule specified.")

    def process_objective(self, objective):
        # we always minimize!
        if objective == "maximize expected power":
            # we minimize the average expected wake loss factor
            self.objective = (
                lambda: self._ds.average_expected_wake_loss_factor)
        elif objective == "minimize cost of energy (Mosetti)":
            # we minimize a proxy for the marginal expected cost of energy per
            # turbine
            self.objective = lambda: (
                _np.exp(-0.00174 * len(self._ds.target) ** 2)
                / (1 - self._ds.average_expected_wake_loss_factor)
            )
        else:
            raise ValueError("Unknown objective specified.")

    def calculate_geometry(self):
        # standard coordinates for vectors
        # between all source and target turbines
        self._ds['vector'] = layout_geometry.generate_vector(
            self._ds.context, self._ds.layout)
        # distances between source and target turbines
        self._ds['distance'] = (
            layout_geometry.generate_distance(self._ds.vector))
        # standard coordinates for unit vectors
        # between all source and target turbines
        self._ds['unit_vector'] = (
            self._ds.vector
            / (self._ds.distance + (self._ds.distance == 0))
        )  # we change 0-distances in the denumerator to 1 to avoid divide by 0

    def calculate_deficit(self):
        # downwind/crosswind coordinates for vectors
        # between all source and target turbines, for all directions
        self._ds['dc_vector'] = layout_geometry.generate_dc_vector(
            self._ds.vector, self._ds.downwind, self._ds.crosswind)
        # deficit
        self._ds['deficit'] = self.wake_model(self._ds.dc_vector
                                              * self.site_radius)
        (self._ds['combined_deficit'],
         self._ds['relative_deficit']) = self.combination_rule(
                                                           self._ds.deficit)

    def expectation(self, array):
        return (array.dot(self._ds.wind_speed_cpmf, 'speed')
                     .dot(self._ds.direction_pmf, 'direction'))

    def calculate_wakeless_power(self):
        self._ds['wakeless_power'] = self.power_curve(self._ds.speed)
        self._ds['expected_wakeless_power'] = self.expectation(
            self._ds.wakeless_power)

    def calculate_power(self):
        # raw values
        power = self.power_curve(
            self._ds.speed * (1 - self._ds.combined_deficit))
        wake_loss = self._ds.wakeless_power - power
        self._ds['wake_loss_factor'] = (
            wake_loss / self._ds.expected_wakeless_power)
        # expectations
        self._ds['expected_wake_loss_factor'] = self.expectation(
            self._ds.wake_loss_factor)
        # turbine average
        self._ds['average_expected_wake_loss_factor'] = (
            self._ds.expected_wake_loss_factor.mean(dim='target'))

    def calculate_relative_wake_loss_vector(self):
        self._ds['relative_wake_loss_vector'] = (
            self._ds.relative_deficit
            * self._ds.wake_loss_factor
            * self._ds.unit_vector
        )

    def calculate_push_down_vector(self):
        return self.expectation(
            self._ds.relative_wake_loss_vector.sum(dim='source')
        )

    def calculate_push_back_vector(self):
        return self.expectation(
            -self._ds.relative_wake_loss_vector.sum(dim='target')
        ).rename(source='target')

    def calculate_push_cross_vector(self):
        # TODO: perhaps dot with everything except 'direction' instead of 'xy'?
        return self.expectation(
            self._ds.relative_wake_loss_vector.sum(dim='source')
                                              .dot(self._ds.crosswind, 'xy')
            * self._ds.crosswind
        )
