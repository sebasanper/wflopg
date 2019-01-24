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


def parcels(parcels_list):
    processed_parcels = []
    for area in parcels_list:
        processed_area = {}
        if 'constraints' in area:
            processed_area['constraints'] = xr.DataArray(
                [[constraint.get(coefficient, 0)
                  for coefficient in COORDS['monomial']]
                 for constraint in area['constraints']],
                dims=['constraint', 'monomial'],
                coords={'monomial': COORDS['monomial']}
            )
            processed_area['constraints'].attrs['rotor_constraint'] = (
                area.get('rotor_constraint', False))
        elif 'circle' in area:
            processed_area['circle'] = xr.DataArray(
                area['circle']['center'], coords=[('xy', COORDS['xy'])])
            processed_area['circle'].attrs['radius'] = (
                area['circle']['radius'])
            processed_area['circle'].attrs['rotor_constraint'] = (
                area.get('rotor_constraint', False))
        else:
            raise ValueError(
                "An area must be described by either constraints or a circle")
        if 'exclusions' in area:
            processed_area['exclusions'] = parcels(area['exclusions'])
        processed_parcels.append(processed_area)

    return processed_parcels
