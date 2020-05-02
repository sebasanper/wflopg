import xarray as xr
import wflopg

# plots wake losses vs. wind sector subdivisions
subs = [1, 2, 3, 4, 5, 9, 12]
n = len(subs)
results = xr.DataArray(np.zeros((n, n)), coords=[('subdivisions', subs), ('layout', subs)])

for subdivisions in results.subdivisions.values:
    for layout in results.layout.values:
        o = wflopg.Owflop()
        o.load_problem('problem-IEA37_CS3-sub{}.yaml'.format(subdivisions),
                       layout='layout-IEA37_CS3-sub{}.yaml'.format(layout))
        o.calculate_geometry()
        o.calculate_wakeless_power()
        o.calculate_deficit()
        o.calculate_power()
        results.loc[{'subdivisions': subdivisions, 'layout': layout}] = o._ds['average_expected_wake_loss_factor']
        
results.plot.line(x='subdivisions')
