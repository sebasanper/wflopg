import numpy as np
import xarray as xr
import pypoman.polygon as ppmp

from wflopg.constants import COORDS


def xy_to_monomial(xy):
    """Return monomial ‘coordinates’ for the given xy-coordinates

    This function works for any xarray DataArray with xy as a dimension.

    """
    monshape = list(xy.shape)
    monshape[xy.dims.index('xy')] = 3
    mon = xr.DataArray(np.ones(monshape),
                       dims=xy.dims, coords={'xy': COORDS['monomial']})
    mon.loc[{'xy': COORDS['xy']}] = xy
    return mon.rename(xy='monomial')


def boundaries(boundaries_list):
    processed_boundaries = []
    for boundary_nesting in boundaries_list:
        processed_boundary_nesting = {}
        if 'polygon' in boundary_nesting:
            processed_boundary_nesting['polygon'] = xr.DataArray(
                boundary_nesting['polygon'],
                dims=['vertex', 'xy'], coords={'xy': COORDS['xy']}
            )
        elif 'circle' in boundary_nesting:
            processed_boundary_nesting['circle'] = xr.DataArray(
                boundary_nesting['circle']['center'],
                coords=[('xy', COORDS['xy'])]
            )
            processed_boundary_nesting['circle'].attrs['radius'] = (
                boundary_nesting['circle']['radius'])
        else:
            raise ValueError(
                "A boundary must be described by either a polygon or a circle")
        if 'exclusions' in boundary_nesting:
            processed_boundary_nesting['exclusions'] = boundaries(
                boundary_nesting['exclusions'])
        processed_boundaries.append(processed_boundary_nesting)

    return processed_boundaries


def parcels(parcels_list, rotor_radius):
    """Return a recursive list of processed parcels

    The parcel list must be of the form described in the site schema.
    The rotor radius must be the site-adimensional rotor radius.

    """
    def parcels_recursive(area, exclusion=True, previous_coeffs=None):
        processed_area = {}
        sign = -1 if exclusion else 1
        if 'constraints' in area:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
            coeffs = xr.DataArray(
                [[constraint.get(coefficient, 0)
                  for coefficient in COORDS['monomial']]
                 for constraint in area['constraints']],
                dims=['constraint', 'monomial'],
                coords={'monomial': COORDS['monomial']}
            )
            norms = np.sqrt(
                np.square(coeffs.sel(monomial=['x', 'y'])).sum(dim='monomial')
            )
            coeffs = coeffs / norms  # normalize the coefficients
            rotor_constraint = xr.DataArray(
                [constraint.get('rotor_constraint', False)
                 for constraint in area['constraints']],
                dims=['constraint']
            )
            safety = sign * rotor_radius * rotor_constraint
            coeffs.loc[{'monomial': '1'}] = (  # include rotor constraint
                coeffs.sel(monomial='1') + safety)
            processed_area['constraints'] = coeffs
            processed_area['border_seeker'] = -sign * coeffs.sel(
                monomial=COORDS['xy']).rename(monomial='xy')
            if exclusion and (previous_coeffs is not None):
                # we must include coefficients of the encompassing constraints
                # (if any) as well to get the vertices
                coeffs = xr.concat([coeffs, previous_coeffs], 'constraint')
                previous_coeffs = None
            else:
                previous_coeffs = coeffs
            vertices = ppmp.compute_polygon_hull(
                coeffs.sel(monomial=COORDS['xy']).values,
                -coeffs.sel(monomial='1').values
            )
            processed_area['vertices'] = xr.DataArray(
                vertices, dims=['vertex', 'xy'], coords={'xy': COORDS['xy']})
            if exclusion:
                # create a mask for vertices that fall outside the site
                vertices_mon = xy_to_monomial(processed_area['vertices'])
                distance = processed_area['constraints'].dot(vertices_mon)
                # for exclusions, turbines with a negative constraint
                # evaluation value violate that constraint
                processed_area['violates'] = (
                    distance < 0).all(dim='constraint')
        elif 'circle' in area:
            processed_area['circle'] = xr.DataArray(
                area['circle']['center'], coords=[('xy', COORDS['xy'])])
            dist = area['circle']['radius']
            if area['circle'].get('rotor_constraint', False):
                dist += -sign * rotor_radius
            processed_area['circle'].attrs['radius'] = dist
            previous_coeffs = None
        else:
            raise ValueError(
                "An area must be described by either constraints or a circle")
        if 'exclusions' in area:
            processed_area['exclusions'] = [
                parcels_recursive(area, not exclusion, previous_coeffs)
                for area in area['exclusions']
            ]

        return processed_area

    parcels_dict = {
        'circle': {'center': [0, 0], 'radius': np.inf},
        'exclusions': parcels_list
    }
    return parcels_recursive(parcels_dict)
