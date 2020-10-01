"""Script to generate WES-paper figures for IEA WT37 CS1 ‘small’."""

import os
import numpy as np
from matplotlib import pyplot as plt

import wflopg
import wflopg.optimizers as opt
import wflopg.visualization as vis


os.chdir("../../../documents/IEA37_CS1+2/")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
plt.rc('axes', titlesize=10)

# %% set up problem
o = wflopg.Owflop()
o.load_problem("problem-IEA37_Optimization_Case_Study-small.yaml")
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
fig.savefig('IEA37C1_16-layout_initial.pdf', bbox_inches="tight")

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
     'scale': 2**-1},
    {'title': 'push-away', 'function': o.calculate_push_away_vector,
     'scale': 2**-1},
    {'title': 'push-back', 'function': o.calculate_push_back_vector,
     'scale': 2**-1},
    {'title': 'push-cross', 'function': o.calculate_push_cross_vector,
     'scale': 2**-6},
]):
    step = pg['function']()
    ax = fig.add_subplot(gs[0, k+2], sharey=alignax)
    if pg['title'] == 'simple':
        alignax = ax
    ax.set_title(pg['title'])
    vis.site_setup(ax, o.minimal_proximity)
    vis.draw_boundaries(ax, o.boundaries)
    vis.draw_turbines(ax, o._ds.layout, o.rotor_radius_adim,
                      turbine_color='#87f')
    vis.draw_step(ax, o._ds.layout, step, o.rotor_radius_adim,
                  scale=pg['scale'])
    ax.set_xlim([-1.5, 1.5])
fig.savefig('IEA37C1_16-wr+pgs.pdf', bbox_inches="tight")

# %% optimize using to various pseudo-gradient types
history = {}
# simple
o = wflopg.Owflop()
o.load_problem("problem-IEA37_Optimization_Case_Study-small.yaml")
o.calculate_wakeless_power()
history['s'] = opt.step_iterator(o, max_iterations=20, methods=['s'])
# push-away
o = wflopg.Owflop()
o.load_problem("problem-IEA37_Optimization_Case_Study-small.yaml")
o.calculate_wakeless_power()
history['a'] = opt.step_iterator(o, max_iterations=20, methods=['a'])
# push-back
o = wflopg.Owflop()
o.load_problem("problem-IEA37_Optimization_Case_Study-small.yaml")
o.calculate_wakeless_power()
history['b'] = opt.step_iterator(o, max_iterations=20, methods=['b'])
# push-cross
o = wflopg.Owflop()
o.load_problem("problem-IEA37_Optimization_Case_Study-small.yaml")
o.calculate_wakeless_power()
history['c'] = opt.step_iterator(o, max_iterations=20, methods=['c'])
# multi-method
o = wflopg.Owflop()
o.load_problem("problem-IEA37_Optimization_Case_Study-small.yaml")
o.calculate_wakeless_power()
history['m'] = opt.step_iterator(o, max_iterations=20)
# %% create and save figure giving overview of optimizations
fig = plt.figure(figsize=(7.7, 4.7))
gs = fig.add_gridspec(ncols=5, nrows=3, wspace=0., hspace=0.,
                      width_ratios=[1] * 5, height_ratios=[1, .9, .8])
axp = {}
shareyp = None
axc = {}
shareyc = None
axs = {}
shareys = None
for k, pg in enumerate({'s': "simple",
                        'a': "push-away", 'b': "push-back", 'c': "push-cross",
                        'm': "multiple"}.items()):
    axp[pg[0]] = fig.add_subplot(gs[0, k], sharey=shareyp)
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
    axp[pg[0]].set_title(pg[1])
    axs[pg[0]] = fig.add_subplot(gs[2, k], sharey=shareys)
    vis.draw_step_size(axs[pg[0]], history[pg[0]])
    axs[pg[0]].set_xlim([-1, 21])
    axs[pg[0]].set_ylim([.011, 2])
    axs[pg[0]].set_xticks([0, 5, 10, 15, 20])
    axc[pg[0]] = fig.add_subplot(gs[1, k], sharex=axs[pg[0]], sharey=shareyc)
    plt.setp(axc[pg[0]].get_xticklabels(), visible=False)
    vis.draw_convergence(axc[pg[0]], history[pg[0]])
    axc[pg[0]].set_xlim([-2.5, 22.5])
    axc[pg[0]].set_ylim([13, 23])
    axc[pg[0]].set_xticks([0, 5, 10, 15, 20])
    axc[pg[0]].set_xlabel('')
    if pg[0] == 's':
        shareyc = axc[pg[0]]
        shareyp = axp[pg[0]]
        shareys = axs[pg[0]]
    else:
        plt.setp(axc[pg[0]].get_yticklabels(), visible=False)
        plt.setp(axs[pg[0]].get_yticklabels(), visible=False)
axc['s'].set_ylabel(r"wake loss [\%]")
axc['s'].set_yticks([14, 16, 18, 20, 22])
axs['s'].set_ylabel("step size [$D$]")
axs['s'].set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
fig.savefig("IEA37C1_16-overview.pdf", bbox_inches="tight")
