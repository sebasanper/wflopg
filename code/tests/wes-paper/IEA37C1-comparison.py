"""Script to generate WES-paper wake loss percentages for IEA WT37 CS1."""

import numpy as np

# wake loss percentages for WES paper optimizations
wlp_WES = {16: 14.0702, 36: 20.726, 64: 21.6277}

# wakeless AEP values for the sites
# (8760 hrs/year * 3.35 (expected wakeless power) * number of turbines)
aep_wakeless = {16: 469536.0, 36: 1056456.0, 64: 1878144.0}

# AEP values from Baker et al. AIAA Scitech 2019 Forum paper
aep = {}
aep[16] = np.array([
    418924.4064,
    414141.2938,
    412251.1945,
    411182.2200,
    409689.4417,
    408360.7813,
    402318.7567,
    392587.8580,
    388758.3573,
    388342.7004,
    366941.5712
])
aep[36] = np.array([
    863676.2993,
    851631.9310,
    849369.7863,
    846357.8142,
    844281.1609,
    828745.5992,
    820394.2402,
    813544.2105,
    777475.7827,
    737883.0985
])
aep[64] = np.array([
    1513311.1936,
    1506388.4151,
    1480850.9759,
    1476689.6627,
    1455075.6084,
    1445967.3772,
    1422268.7144,
    1364943.0077,
    1336164.5498,
    1332883.4328,
    1294974.2977
])

wlp_CS1 = {}
for n in {16, 36, 64}:
    wlp_CS1[n] = 100 * (1 - aep[n]/aep_wakeless[n])

print(wlp_CS1)
