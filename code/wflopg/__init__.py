import numpy as _np
import xarray as _xr
from ruamel.yaml import YAML as _yaml

from wflopg.constants import COORDS
from wflopg import create_turbine
from wflopg import create_site
from wflopg import create_wind
from wflopg import create_wake
from wflopg import create_layout
from wflopg import layout_geometry
from wflopg import create_constraint
from wflopg import create_pam


def _yaml_load(f):
    return _yaml(typ='safe').load(f)


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
                     wind_resource=None, layout=None, wake_model=None,
                     turbine_distance=None):
        """Load wind farm layout optimization problem file

        The file is assumed to be a YAML file in the format described by the
        schema https://bitbucket.org/equaeghe/pseudo_gradients-code/\
        src/master/schemata/wflo_problem-schema.yaml#.

        The keyword arguments make it possible to override problem aspects by
        specifying a file (for `wind_resource` and `layout`) or a `dict` with
        appropriate information (all; see code for what can be done).

        """
        with open(filename) as f:
            problem = _yaml_load(f)
        # extract required parameters directly contained in problem document
        self.problem_uuid = problem['uuid']
        self.turbines = problem['turbines']

        # extract and process information and data from linked documents
        self.load_turbine(problem['turbine'])
        self.load_site(problem['site'])
        self.rotor_radius_adim = self.rotor_radius / self.site_radius
        self.rotor_diameter_adim = 2 * self.rotor_radius_adim
        self.process_turbine()
        self.process_site()
        dir_subs = problem.get('wind_direction_subdivisions', None)
        speeds = problem.get('wind_speeds', None)
        if wind_resource is None:
            wind_resource_file = problem['wind_resource']
        elif isinstance(wind_resource, dict):
            wind_resource_file = wind_resource.get('filename',
                                                   problem['wind_resource'])
            if 'dir_subs' in wind_resource:
                dir_subs = wind_resource['dir_subs']
            if 'speeds' in wind_resource:
                speeds = wind_resource['speeds']
        elif isinstance(wind_resource, str):
            wind_resource_file = wind_resource
        self.load_wind_resource(wind_resource_file)
        self.wind_shear = create_wind.logarithmic_wind_shear(
            self.reference_height, self.roughness_length)
        self.process_wind_resource(
            self.hub_height, self.cut_in, self.cut_out, dir_subs, speeds)

        # process information for wake model-related properties
        if wake_model is None:
            wake_model = problem['wake_model']
        self.process_wake_model(wake_model)
        self.process_objective(problem['objective'])

        # create function to generate turbine constraint violation fixup steps
        if turbine_distance is None:
            turbine_distance = problem.get('turbine_distance', 1)
        self.minimal_proximity = (turbine_distance * self.rotor_diameter_adim)
        self.proximity_repulsion = (
            create_constraint.distance(self.minimal_proximity))

        # deal with initial layout
        if layout is None:
            self.load_layout(problem['layout'])
        elif isinstance(layout, str):
            self.load_layout(layout)
        elif isinstance(layout, _xr.DataArray):
            self.process_initial_layout(layout)
        elif isinstance(layout, dict):
            if 'type' not in layout:
                raise ValueError("Layout type has not been specified!")
            else:
                if layout['type'] == 'hex':
                    turbines = layout.get('turbines', self.turbines)
                    if isinstance(turbines, list):
                        turbines = _np.random.randint(*self.turbines)
                    self.process_initial_layout(
                        create_layout.hexagonal(
                            turbines,
                            self.site_parcels,
                            layout.get('site_violation_distance', 0),
                            self.to_border
                        )
                    )
                elif layout['type'] == 'pam':
                    self._ds['layout'] = create_pam.layout(
                        self.minimal_proximity,
                        layout.get('num_dists', 100),
                        layout.get('num_dirs', 360)
                    )
                    self._ds['context'] = create_pam.context()
                else:
                    raise ValueError("Unknown layout type: "
                                     "‘{}’".format(layout['type']))

    def load_turbine(self, filename):
        with open(filename) as f:
            turbine = _yaml_load(f)
        self.rotor_radius = turbine['rotor_radius']
        self.hub_height = turbine['hub_height']
        self.rated_power = turbine['rated_power']
        self.rated_speed = turbine.get('rated_wind_speed', _np.nan)
        self.cut_in = turbine.get('cut_in', 0.0)
        self.cut_out = turbine.get('cut_out', _np.inf)
        if 'power_curve' in turbine:
            pc = _np.array(turbine['power_curve'])
            order = pc[:, 0].argsort()
            self.power_curve_data = _xr.DataArray(
                pc[order, 1], coords=[('speed', pc[order, 0])])
        else:
            self.power_curve_data = None
        if 'thrust_coefficient' in turbine:
            self.thrust_coefficient = turbine['thrust_coefficient']
            self.thrust_curve = None
        elif 'thrust_curve' in turbine:
            tc = _np.array(turbine['thrust_curve'])
            order = tc[:, 0].argsort()
            self.thrust_curve_data = _xr.DataArray(
                tc[order, 1], coords=[('speed', tc[order, 0])])
            self.thrust_coefficient = None
        else:
            raise ValueError("Turbine document should contain either "
                             "a 'thrust_curve' or "
                             "a constant 'thrust_coefficient'")

    def process_turbine(self):
        # define power curve
        if self.power_curve_data is not None:
            self.power_curve = create_turbine.interpolated_power_curve(
                self.rated_power, self.rated_speed, self.cut_in, self.cut_out,
                self.power_curve_data
            )
        else:  # cubic power curve
            self.power_curve = create_turbine.cubic_power_curve(
                 self.rated_power, self.rated_speed, self.cut_in, self.cut_out)
        # define thrust curve
        if self.thrust_curve is not None:
            self.thrust_curve = create_turbine.interpolated_thrust_curve(
                self.cut_in, self.cut_out, self.thrust_curve_data)
        elif self.thrust_coefficient is not None:  # constant thrust curve
            self.thrust_curve = create_turbine.constant_thrust_curve(
                self.cut_in, self.cut_out, self.thrust_coefficient)

    def load_site(self, filename):
        with open(filename) as f:
            site = _yaml_load(f)
        self.roughness_length = site.get('roughness', None)
        self.site_radius = site['radius'] * 1e3  # km to m
        if 'location' in site:
            if 'utm' in site['location']:
                self.site_location = _xr.DataArray(
                    site['location']['utm'], coords=[('xy', COORDS['xy'])])
            elif 'adhoc' in site['location']:
                self.site_location = _xr.DataArray(
                    site['location']['adhoc'], coords=[('xy', COORDS['xy'])]
                ) * 1e3  # km to m
        self.site_parcels = site['parcels']
        self.site_boundaries = site.get('boundaries', None)
        
    def process_site(self):
        # Process parcels
        self.parcels = create_site.parcels(self.site_parcels,
                                           self.rotor_radius_adim)
        # create function that reports whether a turbine is inside the site
        self.inside = create_constraint.inside_site(self.parcels)
        # create function to generate site constraint violation fixup steps
        self.to_border = create_constraint.site(self.parcels)
        # Process boundaries
        if self.site_boundaries is not None:
            self.boundaries = create_site.boundaries(self.site_boundaries)

    def load_wind_resource(self, filename):
        with open(filename) as f:
            wind_resource = _yaml_load(f)
        self.reference_height = wind_resource['reference_height']
        self.air_density = wind_resource.get('air_density', 1.2041)
        self.atmospheric_stability = wind_resource.get(
            'atmospheric_stability', None)
        self.turbulence_intensity = wind_resource.get(
            'turbulence_intensity', None)

        wind_rose = wind_resource['wind_rose']
        dirs = _np.array(wind_rose['directions'])
        # we do not assume the wind directions are sorted in the data file and
        # therefore sort them here
        order = dirs.argsort()
        dirs_sorted = dirs[order]
        self.wind_rose = _xr.Dataset()
        self.wind_rose['dir_weights'] = _xr.DataArray(
            _np.array(wind_rose['direction_pmf'])[order],
            coords=[('direction', dirs_sorted)]
        )
        if 'speed_cweibull' in wind_rose:
            self.wind_rose['speed_cweibull'] = _xr.DataArray(
                _np.array(wind_rose['speed_cweibull'])[order, :],
                coords=[('direction', dirs_sorted),
                        ('weibull_param', COORDS['weibull_param'])]
            )
        elif 'speed_cpmf' in wind_rose and 'speeds' in wind_rose:
            self.wind_rose['speed_cpmf'] = _xr.DataArray(
                _np.array(wind_rose['speed_cpmf'])[order, :],
                coords=[('direction', dirs_sorted),
                        ('speed', wind_rose['speeds'])]
            )
        else:
            raise ValueError(
                "A conditional wind speed probability distribution "
                "should be given either as parameters for conditional Weibull "
                "distributions or as a conditional probability mass function.")

    def process_wind_resource(self, hub_height, cut_in, cut_out,
                              dir_subs=None, speeds=None):
        # Create the conditional wind speed probability mass function
        #
        # NOTE: Of the conditional wind speed probability mass function, only
        #       the values within the [cut_in, cut_out] interval are stored in
        #       self._ds['wind_speed_cpmf'] (defined below), as the others give
        #       no contribution to the power production.
        #
        if 'speed_cweibull' in self.wind_rose:
            if speeds is None:
                raise ValueError(
                    "An array of wind speeds must be specified in case the "
                    "wind resource is formulated in terms of Weibull "
                    "distributions")
            # take wind shear into account
            speeds = self.wind_shear(hub_height, _np.array(speeds))
            speed_probs = create_wind.discretize_weibull(
                self.wind_rose['speed_cweibull'], cut_in, cut_out, speeds)
        elif 'speed_cpmf' in self.wind_rose:
            # take wind shear into account
            if speeds is None:
                speeds = self.wind_shear(
                    hub_height,
                    self.wind_rose['speed_cpmf'].coords['speed'].values
                )
            speed_probs = create_wind.conformize_cpmf(
                self.wind_rose['speed_cpmf'], cut_in, cut_out, speeds)

        # Subdivide wind direction and speed pmfs if needed
        dir_weights = self.wind_rose['dir_weights']
        if dir_subs:
            dir_weights, speed_probs = create_wind.subdivide(
                dir_weights, speed_probs, dir_subs)

        # Store pmfs
        self._ds['direction_pmf'] = dir_weights / dir_weights.sum()
        self._ds['wind_speed_cpmf'] = speed_probs

        # Store downwind and crosswind unit vectors
        self._ds['downwind'] = layout_geometry.generate_downwind(
            self._ds.direction)
        self._ds['crosswind'] = layout_geometry.generate_crosswind(
            self._ds.downwind)

    def process_wake_model(self, model):
        thrusts = self.thrust_curve(self._ds.speed)

        # preliminaries for wake model definition
        wake_type = model.get('wake_type', "linear top hat")
        expansion_coeff = model.get('expansion_coefficient', None)
        stream_tube_assumption = model.get('stream_tube_assumption', "rotor")
        deficit_type = model.get('deficit_type', "Jensen")
        averaging = model.get('averaging', False)

        # define wake model
        if wake_type == "linear top hat":
            if not expansion_coeff:
                if deficit_type == "Jensen":
                    if self.roughness_length is None:
                        expansion_coeff = 0.067
                        # From doi:10.1088/1742-6596/1037/7/072019
                        # caption of Fig.3
                    else:
                        expansion_coeff = (
                            0.5 / _np.log(self.hub_height
                                          / self.roughness_length))
                if deficit_type == "Frandsen":
                    expansion_coeff = 0.027
                    # From doi:10.1088/1742-6596/1037/7/072019 caption of Fig.3
            self.wake_model = create_wake.linear_top_hat(
                thrusts, self.rotor_radius, expansion_coeff,
                deficit_type, stream_tube_assumption, averaging
            )
        elif wake_type == "entrainment":
            self.wake_model = create_wake.entrainment(
                thrusts, self.rotor_radius, averaging)
        elif wake_type == "BPA (IEA37)":
            self.wake_model = create_wake.bpa_iea37(
                thrusts, self.rotor_radius, self.turbulence_intensity)
        else:
            raise ValueError(
                "Unknown wake type specified: ‘{}’.".format(wake_type))

        # define combination rule
        combination_rule = model.get('combination', "RSS")
        if combination_rule == "RSS":
            self.combination_rule = create_wake.rss_combination()
        elif combination_rule == "NO":
            self.combination_rule = create_wake.no_combination()
        else:
            raise ValueError("Unknown combination rule specified: "
                             "‘{}’.".format(combination_rule))

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

    def load_layout(self, filename):
        with open(filename) as f:
            self.process_initial_layout(_yaml_load(f)['layout'])

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
        self._ds['deficit'] = self.wake_model(
            self._ds.dc_vector * self.site_radius)
        self._ds['combined_deficit'], self._ds['relative_deficit'] = (
            self.combination_rule(self._ds.deficit))

    def conditional_expectation_wind_speed(self, array):
        return array.dot(self._ds.wind_speed_cpmf, 'speed')

    def expectation_direction(self, array):
        return array.dot(self._ds.direction_pmf, 'direction')

    def expectation(self, array):
        return self.expectation_direction(
            self.conditional_expectation_wind_speed(array))

    def calculate_wakeless_power(self):
        self._ds['wakeless_power'] = self.power_curve(self._ds.speed)
        self._ds['expected_wakeless_power'] = self.expectation(
            self._ds.wakeless_power)

    def calculate_power(self):
        # raw values
        self._ds['power'] = self.power_curve(
            self._ds.speed * (1 - self._ds.combined_deficit))
        wake_loss = self._ds.wakeless_power - self._ds.power
        self._ds['wake_loss_factor'] = (
            wake_loss / self._ds.expected_wakeless_power)
        # expectations
        self._ds['expected_wake_loss_factor'] = self.expectation(
            self._ds.wake_loss_factor)
        # turbine average
        self._ds['average_expected_wake_loss_factor'] = (
            self._ds.expected_wake_loss_factor.mean(dim='target'))

    def calculate_aep(self):
        hrs_per_yr = 8760
        dir_power = self.conditional_expectation_wind_speed(
            self._ds.power.sum(dim='target'))
        self._ds['dir_AEP'] = hrs_per_yr * dir_power * self._ds.direction_pmf
        self._ds['AEP'] = self._ds['dir_AEP'].sum(dim='direction')

    def calculate_relative_wake_loss_vector(self):
        self._ds['relative_wake_loss_vector'] = (
            self._ds.relative_deficit
            * self._ds.wake_loss_factor
            * self._ds.unit_vector
        )

    def calculate_push_away_vector(self):
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
