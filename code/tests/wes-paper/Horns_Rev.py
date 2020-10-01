"""Script to generate WES-paper figures for Horns Rev as per Feng & Shen."""

import os
import numpy as np
from matplotlib import pyplot as plt

import wflopg
import wflopg.optimizers as opt
import wflopg.visualization as vis


os.chdir("../../../documents/Horns_Rev/")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
plt.rc('axes', titlesize=10)

# %% set up problem
o = wflopg.Owflop()
o.load_problem("problem-Horns_Rev.yaml",
               wind_resource={'interpolation': 'nearest'})
o.calculate_wakeless_power()
o.calculate_deficit()
o.calculate_power()
o.calculate_relative_wake_loss_vector()

# %% save initial layout figure
fig = plt.figure()
ax = plt.subplot(111)
vis.site_setup(ax, o.minimal_proximity)
vis.draw_boundaries(ax, o.boundaries)
vis.draw_turbines(ax, o._ds.layout, o.rotor_radius_adim, o.minimal_proximity)
fig.savefig('Horns_Rev-layout_initial.pdf', bbox_inches="tight")

# %% optimize using to various pseudo-gradient types
history = {}
# %% start from Mosetti's layout
o = wflopg.Owflop()
o.load_problem("problem-Horns_Rev.yaml",
               wind_resource={'interpolation': 'nearest'})
o.calculate_wakeless_power()
history['M'] = opt.step_iterator(o, max_iterations=30,
                                 multiplier=0.5, scaling=[.7, 1.1])
# %% start from hex layout
o = wflopg.Owflop()
o.load_problem(
    "problem-Horns_Rev.yaml",
    wind_resource={'interpolation': 'nearest'},
    layout={
        'type': 'hex', 'kwargs': {'offset': [0.5, 0], 'angle': np.radians(15)}}
)
o.calculate_wakeless_power()
history['H'] = opt.step_iterator(o, max_iterations=30,
                                 multiplier=2, scaling=[.8, 1.1])
# %% create and save figure giving overview of optimizations
fig = plt.figure(figsize=(3.4, 5.6))
gs = fig.add_gridspec(ncols=2, nrows=5, wspace=0., hspace=0.,
                      width_ratios=[1] * 2, height_ratios=[1, .4, 1.3, 1, 1])
#
axwr = fig.add_subplot(gs[0, :], polar=True)
vis.draw_windrose(axwr, o._ds.direction_pmf)
axwr.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
axwr.set_xticklabels(['N', 'O', 'S', 'W'])
axwr.xaxis.set_tick_params(pad=-2)
axwr.set_rticks([])
#
axp = {}
shareyp = None
axc = {}
shareyc = None
axs = {}
shareys = None
for k, pg in enumerate({'M': "Horns Rev", 'H': "hexagonal"}.items()):
    axp[pg[0]] = fig.add_subplot(gs[2, k], sharey=shareyp)
    vis.site_setup(axp[pg[0]], o.minimal_proximity)
    vis.draw_boundaries(axp[pg[0]], o.boundaries)
    vis.draw_turbines(axp[pg[0]], history[pg[0]].layout.sel(iteration=0),
                      o.rotor_radius_adim, turbine_color='#87f')
    vis.draw_turbines(
        axp[pg[0]],
        history[pg[0]].layout.isel(
            iteration=history[pg[0]].objective.argmin()),
        o.rotor_radius_adim,
        o.minimal_proximity,
        turbine_color='#f56'
    )
    vis.connect_layouts(axp[pg[0]], history[pg[0]].layout)
    axp[pg[0]].set_xlim([-np.sqrt(.5)-.2, np.sqrt(.5)+.2])
    axp[pg[0]].set_ylim([-np.sqrt(.5)-.2, np.sqrt(.5)+.2])
    axp[pg[0]].set_title(pg[1])
    axs[pg[0]] = fig.add_subplot(gs[4, k], sharey=shareys)
    vis.draw_step_size(axs[pg[0]], history[pg[0]])
    axs[pg[0]].set_xlim([-2.5, 322.5])
    # axs[pg[0]].set_ylim([.004, 14])
    axs[pg[0]].set_xticks([0, 5, 10, 15, 20, 25, 30])
    axc[pg[0]] = fig.add_subplot(gs[3, k], sharex=axs[pg[0]], sharey=shareyc)
    plt.setp(axc[pg[0]].get_xticklabels(), visible=False)
    vis.draw_convergence(axc[pg[0]], history[pg[0]])
    axc[pg[0]].set_xlim([-2.5, 32.5])
    # axc[pg[0]].set_ylim([1.5, 5])
    axc[pg[0]].set_xticks([0, 5, 10, 15, 20, 25, 30])
    axc[pg[0]].set_xlabel('')
    if pg[0] == 'M':
        shareyc = axc[pg[0]]
        shareyp = axp[pg[0]]
        shareys = axs[pg[0]]
    else:
        plt.setp(axc[pg[0]].get_yticklabels(), visible=False)
        plt.setp(axs[pg[0]].get_yticklabels(), visible=False)
axc['M'].set_ylabel(r"wake loss [\%]")
axc['M'].set_yticks([8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75])
axc['M'].set_yticklabels(['', '8.5', '', '9.0', '', '9.5', ''])
axs['M'].set_ylabel("step size [$D$]")
axs['M'].set_yticks([0.1, 1])
axs['M'].set_yticklabels([0.1, 1])
fig.savefig("Horns_Rev-overview.pdf", bbox_inches="tight")
