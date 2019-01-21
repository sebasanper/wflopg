import numpy as np
import xarray as xr


coefficients = ['1', 'x', 'y', 'xy', 'xx', 'yy']


def boundaries(boundaries_list):
    processed_boundaries = []
    for boundary_nesting in boundaries_list:
        processed_boundary_nesting = {}
        if 'boundary' in boundary_nesting:
            processed_boundary_nesting['boundary'] = xr.DataArray(
                boundary_nesting['boundary'],
                dims=['vertex', 'xy_coord'], coords={'xy_coord': ['x', 'y']}
            )
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
                  for coefficient in coefficients]
                 for constraint in area['constraints']],
                dims=['constraint', 'coefficient'],
                coords={'coefficient': coefficients}
            )
        if 'exclusions' in area:
            processed_area['exclusions'] = parcels(area['exclusions'])
        processed_parcels.append(processed_area)

    return processed_parcels
