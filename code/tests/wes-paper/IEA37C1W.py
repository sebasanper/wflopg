"""Script to generate WES-paper figures for IEA WT37 CS1 ‘from West’."""

import os
import numpy as np
from matplotlib import pyplot as plt

import wflopg
import wflopg.visualization as vis


os.chdir("../../../documents/IEA37_CS1+2/")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
plt.rc('axes', titlesize=10)

# %% set up problem
o = wflopg.Owflop()
o.load_problem("problem-IEA37_Optimization_Case_Study-small.yaml",
               wind_resource="wind_resource-from_West.yaml")
o.calculate_wakeless_power()
o.calculate_deficit()
o.calculate_power()
o.calculate_relative_wake_loss_vector()

# %% save initial layout figure
fig = plt.figure()
ax = plt.subplot(111)
vis.site_setup(ax, o.minimal_proximity)
vis.draw_boundaries(ax, o.site_boundaries)
vis.draw_turbines(ax, o._ds.layout, o.rotor_radius_adim, o.minimal_proximity)
fig.savefig('IEA37C1W_16-layout_initial.pdf', bbox_inches="tight")

# %% plot windrose and initial pseudogradients
fig = plt.figure(figsize=(8, 1.7))
alignax = None
gs = fig.add_gridspec(ncols=6, nrows=1, wspace=0., hspace=0.,
                      width_ratios=[.8, .2] + 4*[1])
axwr = fig.add_subplot(gs[0:2, 0], polar=True)
vis.draw_windrose(axwr, o.wind_rose.dir_weights)
axwr.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
axwr.set_xticklabels(['N', 'O', 'S', 'W'])
axwr.xaxis.set_tick_params(pad=-2)
axwr.set_rticks([])
#
for k, pg in enumerate([
    {'title': 'simple', 'function': o.calculate_simple_vector,
     'scale': 2**+1},
    {'title': 'push-away', 'function': o.calculate_push_away_vector,
     'scale': 2**+1},
    {'title': 'push-back', 'function': o.calculate_push_back_vector,
     'scale': 2**+1},
    {'title': 'push-cross', 'function': o.calculate_push_cross_vector,
     'scale': 2**-4},
]):
    step = pg['function']()
    ax = fig.add_subplot(gs[0, k+2], sharey=alignax)
    if pg['title'] == 'simple':
        alignax = ax
    ax.set_title(pg['title'])
    vis.site_setup(ax, o.minimal_proximity)
    vis.draw_boundaries(ax, o.boundaries)
    vis.draw_turbines(ax, o._ds.layout, o.rotor_radius_adim,
                      turbine_color='gray')
    vis.draw_step(ax, o._ds.layout, step, o.rotor_radius_adim,
                  scale=pg['scale'])
    ax.set_xlim([-1.5, 1.5])
fig.savefig('IEA37C1W_16-wr+pgs.pdf', bbox_inches="tight")
