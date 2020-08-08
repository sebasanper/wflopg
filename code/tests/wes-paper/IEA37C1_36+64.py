"""Script to generate WES-paper figures for IEA WT37 CS1 ‘medium’/‘large’."""

import os
from matplotlib import pyplot as plt

import wflopg
import wflopg.optimizers as opt
import wflopg.visualization as vis


os.chdir("../../../documents/IEA37_CS1+2/")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
plt.rc('axes', titlesize=10)

# %% set-up and optimize (generate problem objects and histories)
os = {}
histories = {}
for n, params in {
    36: {'size': "medium", 'iterations': 30,
         'kwargs': {'multiplier': 1.3, 'scaling': [.5, .99]}},
    64: {'size': "large", 'iterations': 30,
         'kwargs': {'multiplier': 3, 'scaling': [.9, 2]}}
}.items():
    os[n] = wflopg.Owflop()
    os[n].load_problem(
        f"problem-IEA37_Optimization_Case_Study-{params['size']}.yaml")
    os[n].calculate_wakeless_power()
    histories[n] = opt.step_iterator(os[n],
                                     max_iterations=params['iterations'],
                                     **params['kwargs'])

# %% create and save figure giving overview of optimizations
fig = plt.figure(figsize=(3.4, 5.))
gs = fig.add_gridspec(ncols=2, nrows=3, wspace=0., hspace=0.,
                      width_ratios=[1] * 2, height_ratios=[1.1, .9, .9])
axp = {}
shareyp = None
axc = {}
shareyc = None
axs = {}
shareys = None
for k, n in enumerate([36, 64]):
    axp[n] = fig.add_subplot(gs[0, k], sharey=shareyp)
    vis.site_setup(axp[n], os[n].minimal_proximity)
    vis.draw_boundaries(axp[n], os[n].boundaries)
    vis.draw_turbines(axp[n], histories[n].layout.sel(iteration=0),
                      os[n].rotor_radius_adim, turbine_color='gray')
    vis.draw_turbines(
        axp[n],
        histories[n].layout.isel(
            iteration=histories[n].objective.argmin()),
        os[n].rotor_radius_adim,
        os[n].minimal_proximity,
        turbine_color='b'
    )
    vis.connect_layouts(axp[n], histories[n].layout)
    axp[n].set_title(f"{n} turbines")
    axs[n] = fig.add_subplot(gs[2, k], sharey=shareys)
    vis.draw_step_size(axs[n], histories[n])
    axs[n].set_xlim([-2.5, 32.5])
    axs[n].set_ylim([.009, 15])
    axs[n].set_xticks([0, 5, 10, 15, 20, 25, 30])
    axc[n] = fig.add_subplot(gs[1, k], sharex=axs[n], sharey=shareyc)
    plt.setp(axc[n].get_xticklabels(), visible=False)
    vis.draw_convergence(axc[n], histories[n])
    axc[n].set_xlim([-2.5, 32.5])
    axc[n].set_ylim([19, 32.5])
    axc[n].set_xticks([0, 5, 10, 15, 20, 25, 30])
    axc[n].set_xlabel('')
    if n == 36:
        shareyc = axc[n]
        shareyp = axp[n]
        shareys = axs[n]
    else:
        plt.setp(axc[n].get_yticklabels(), visible=False)
        plt.setp(axs[n].get_yticklabels(), visible=False)
axc[36].set_ylabel(r"wake loss [\%]")
axc[36].set_yticks([20, 22, 24, 26, 28, 30, 32])
axs[36].set_ylabel("step size [$D$]")
axs[36].set_yticks([0.01, 0.1, 1, 10])
axs[36].set_yticklabels(['0.01', '0.1', '1', '10'])
fig.savefig("IEA37C1_36+64-overview.pdf", bbox_inches="tight")
