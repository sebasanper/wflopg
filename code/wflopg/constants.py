# Dimensions for xarray DataSet and DataArrays
# Coordinates that are universal over all problems are defined in COORDS below
# Problem-specific coordinates are defined in the code
DIMS = {
    # Cartesian coordinates, where ‘x’ corresponds to the South-North direction
    # and ‘y’ to the West-East direction."""
    'xy',
    # Cartesian coordinates determined by a given wind direction, where ‘d’
    # corresponds to the downwind direction and ‘c’ to the crosswind direction.
    'dc',
    # Labels for the coefficients of the monomials in quadratic expressions, or
    # for the values of these monomials.
    'quad',
    # Lower and upper bounds of an interval
    'interval',
    # Weibull parameters
    'weibull_param',
    # Wind direction
    'direction',
    # Wind speed
    'speed',
    # Turbines that are a source of wakes (for now the same as 'target')
    'source',
    # Turbines that are a target of wakes (for now the same as 'source')
    'target',
    # Vertices of boundaries (differs from boundary to boundary)
    'vertex',
    # Constraints of parcels (differs from parcel to parcel)
    'constraint'
}

# Universal coordinates for xarray DataSet and DataArrays
COORDS = {
    'xy': ['x', 'y'],
    'dc': ['d', 'c'],
    'quad': ['1', 'x', 'y', 'xy', 'xx', 'yy'],
    'interval': ['lower', 'upper'],
    'weibull_param': ['scale', 'shape']
}
