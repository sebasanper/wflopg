"""Script to generate WES-paper figures for self-created BWFS IV site."""

import os
import numpy as np
from matplotlib import pyplot as plt

import wflopg
import wflopg.optimizers as opt
import wflopg.visualization as vis


os.chdir("../../../documents/BWFS/")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
plt.rc('axes', titlesize=10)

# %% save wind rose figure
o = wflopg.Owflop()
o.load_problem(
    "problem-BWFS-IV-WES.yaml",
    wind_resource={'interpolation': 'linear'},
    layout={'type': 'hex', 'turbines': 3}
)
fig = plt.figure(figsize=(1.28, 1.7))
axwr = fig.add_subplot(111, polar=True)
vis.draw_windrose(axwr, o._ds.direction_pmf)
axwr.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
axwr.set_xticklabels(['N', 'O', 'S', 'W'])
axwr.xaxis.set_tick_params(pad=-2)
axwr.set_rticks([])
fig.savefig("Borssele_IV-wind_rose.pdf", bbox_inches="tight")


# %% optimize using different parameters and turbine numbers
history = {}
cases = {30: ([4, 4.5, 5, 5.5], ['4', '', '5', '']),
         50: ([7.5, 8.0, 8.5], ['7.5', '8.0', '8.5']),
         70: ([10, 10.5, 11, 11.5, 12, 12.5], ['10', '', '11', '', '12', '']),
         90: ([13.5, 14, 14.5, 15, 15.5], ['', '14', '', '15', ''])}
for kind, multiplier, scaling in [('soft', 3, [0.9, 1.1]),
                                  ('aggressive', 3, [.5, 2])]:
    for turbines in cases:
        o = wflopg.Owflop()
        o.load_problem(
            "problem-BWFS-IV-WES.yaml",
            wind_resource={'interpolation': 'linear'},
            layout={
                'type': 'hex', 'turbines': turbines,
                'kwargs': {'offset': [0, 0], 'angle': np.radians(13)}}
        )
        o.calculate_wakeless_power()
        history[(kind, turbines)] = opt.step_iterator(
            o, max_iterations=15, multiplier=multiplier, scaling=scaling)

# %% create and save figure giving overview of optimizations
for turbines in cases:
    fig = plt.figure(figsize=(3.4, 4.4))
    gs = fig.add_gridspec(ncols=2, nrows=3, wspace=0., hspace=0.,
                          width_ratios=[1] * 2, height_ratios=[1.1, 1, 1])
    #
    axp = {}
    shareyp = None
    axc = {}
    shareyc = None
    axs = {}
    shareys = None
    for k, kind in enumerate(['soft', 'aggressive']):
        axp[kind] = fig.add_subplot(gs[0, k], sharey=shareyp)
        vis.site_setup(axp[kind], o.minimal_proximity)
        vis.draw_boundaries(axp[kind], o.boundaries)
        vis.draw_turbines(axp[kind],
                          history[(kind, turbines)].layout.sel(iteration=0),
                          o.rotor_radius_adim, turbine_color='gray')
        vis.draw_turbines(
            axp[kind],
            history[(kind, turbines)].layout.isel(
                iteration=history[(kind, turbines)].objective.argmin()),
            o.rotor_radius_adim,
            o.minimal_proximity,
            turbine_color='b'
        )
        vis.connect_layouts(axp[kind], history[(kind, turbines)].layout)
        axp[kind].set_xlim([-1, 1])
        axp[kind].set_ylim([-1, .5])
        axp[kind].set_title(kind)
        axs[kind] = fig.add_subplot(gs[2, k], sharey=shareys)
        vis.draw_step_size(axs[kind], history[(kind, turbines)])
        axs[kind].set_xlim([-2.5, 17.5])
        axs[kind].set_ylim([.02, 50])
        axs[kind].set_xticks([0, 5, 10, 15])
        axc[kind] = fig.add_subplot(gs[1, k], sharex=axs[kind], sharey=shareyc)
        plt.setp(axc[kind].get_xticklabels(), visible=False)
        vis.draw_convergence(axc[kind], history[(kind, turbines)])
        axc[kind].set_xlim([-2.5, 17.5])
        # axc[kind].set_ylim([1.5, 5])
        axc[kind].set_xticks([0, 5, 10, 15])
        axc[kind].set_xlabel('')
        if kind == 'soft':
            shareyc = axc[kind]
            shareyp = axp[kind]
            shareys = axs[kind]
        else:
            plt.setp(axc[kind].get_yticklabels(), visible=False)
            plt.setp(axs[kind].get_yticklabels(), visible=False)
    axc['soft'].set_ylabel(r"wake loss [\%]")
    axc['soft'].set_yticks(cases[turbines][0])
    axc['soft'].set_yticklabels(cases[turbines][1])
    axs['soft'].set_ylabel("step size [$D$]")
    axs['soft'].set_yticks([.1, 1, 10])
    axs['soft'].set_yticklabels(['0.1', '1', '10'])
    fig.savefig(f"Borssele_IV-overview-{turbines}.pdf", bbox_inches="tight")
