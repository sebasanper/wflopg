import numpy as np
import xarray as xr

from wflopg.constants import COORDS


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
    def parcels_recursive(area, exclusion=True):
        processed_area = {}
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
                np.square(coeffs.sel(monomial='x', drop=True))
                + np.square(coeffs.sel(monomial='y', drop=True))
            )
            coeffs = coeffs / norms  # normalize the coefficients
            rotor_constraint = xr.DataArray(
                [constraint.get('rotor_constraint', False)
                 for constraint in area['constraints']],
                dims=['constraint']
            )
            safety = xr.where(rotor_constraint, rotor_radius, 0)
            coeffs.loc[{'monomial': '1'}] = (  # include rotor constraint
                coeffs.sel(monomial='1') + safety)
            processed_area['constraints'] = coeffs
            processed_area['border_seeker'] = -coeffs.sel(
                monomial=COORDS['xy']).rename(monomial='xy')
            vertices_hom = np.cross(
                coeffs.values, np.roll(coeffs.values, 1, axis=0))
            processed_area['vertices'] = xr.DataArray(
                vertices_hom[:,1:] / vertices_hom[:,:1],
                dims=['constraint', 'xy'], coords={'xy': COORDS['xy']}
            )
        elif 'circle' in area:
            processed_area['circle'] = xr.DataArray(
                area['circle']['center'], coords=[('xy', COORDS['xy'])])
            dist = area['circle']['radius']
            if area['circle'].get('rotor_constraint', False):
                dist += rotor_radius if exclusion else -rotor_radius
            processed_area['circle'].attrs['radius_sqr'] = np.square(dist)
        else:
            raise ValueError(
                "An area must be described by either constraints or a circle")
        if 'exclusions' in area:
            processed_area['exclusions'] = [
                parcels_recursive(area, not exclusion)
                for area in area['exclusions']
            ]

        return processed_area

    parcels_dict = {
        'circle': {'center': [0, 0], 'radius': np.inf},
        'exclusions': parcels_list
    }
    return parcels_recursive(parcels_dict)
