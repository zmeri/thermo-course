'''
Näide sellest, kuidas leida PC-SAFT olekuvõrrandi parameetrid andmetest
'''
import numpy as np
import pandas as pd
from pcsaft import pcsaft_den, flashTQ, SolutionError # dokumentatsioon: https://pcsaft.readthedocs.io/en/latest/
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

import datetime

# impordi andmed ---------
data = pd.read_csv('andmed_1-hekseen.csv', header=2, delimiter=';')

# kontrolli andmeid joonistega ---------
plt.figure()
plt.scatter(data.loc[data['omadus'] == 'tihedus', 'temperatuur'],
            data.loc[data['omadus'] == 'tihedus', 'tihedus_molaarne'])
plt.title('1-hekseen')
plt.xlabel('Temperatuur (K)')
plt.ylabel('Tihedus (mol/m$^3$)')
plt.savefig('1hekseen_andmete_kontroll_den', dpi=400)
plt.show()

plt.figure()
plt.scatter(data.loc[data['omadus'] == 'aururõhk', 'temperatuur'],
            data.loc[data['omadus'] == 'aururõhk', 'rõhk'])
plt.yscale('log')
plt.title('1-hekseen')
plt.xlabel('Temperatuur (K)')
plt.ylabel('Aururõhk (Pa)')
plt.savefig('1hekseen_andmete_kontroll_vp', dpi=400)
plt.show()


# defineerime funktsiooni, mida saame optimeerida
def cost_pcsaft(params, omadus, t, p, den):
    cost = 0
    x = np.asarray([1]) # puhas aine, siis moolosa on alati 1
    # optimeerimise funktsioon annab parameetrid Numpy massiivina. Siin
    # muudame sõnastikuks, nagu pcsaft funktsioonid soovivad.
    params_pcsaft = {'m':params[0], 's':params[1], 'e':params[2]}

    # pcsaft paketiga peame tegema arvutust ühele punktile korraga. Sellepärast kasutame for-tsüklit
    for i in range(t.shape[0]):
        if omadus[i] == 'aururõhk':
            try:
                p_fit = flashTQ(t[i], 0, x, params_pcsaft)[0]
                cost = cost + ((p_fit - p[i]) / p[i] * 100)**2
            except:
                cost = cost + 1e20 # kui ei leia head lahendust, siis lihtsalt lisame suure numbri
        elif omadus[i] == 'tihedus':
            den_fit = pcsaft_den(t[i], p[i], x, params_pcsaft, phase='liq')
            cost = cost + ((den_fit - den[i]) / den[i] * 100)**2

    # kontrollime, et viga ei ole NAN (not a number). Kui on NAN, siis lahendaja ei
    # käitu hästi. Mõnikord juhtub näiteks siis, kui proovitud parameetrid on väga valed.
    if not np.isfinite(cost):
        cost = 1e60
    return cost


# leia PC-SAFT parameetrid ---------
bnds = ((2,7), (2,6), (150,550)) # Siin paneme piirid paika. Mis piirides parameetrid tõenäoliselt on?
print('Leian kij lahendajaga. See võib võtta mitu minutit.')
result = differential_evolution(cost_pcsaft, bounds=bnds,
                                args=(data['omadus'], data['temperatuur'], data['rõhk'], data['tihedus_molaarne']))
# Kui su arvutil on mitu tuuma võid kasutada mitu korraga, et optimeerimine oleks kiirem. Anna
# differential_evolution funktsioonile lisa parameeter "workers". Kui tahad kasutada
# kõik tuumad pane väärtuseks -1.
# result = differential_evolution(cost_pcsaft, bounds=bnds, workers=-1,
#                                 args=(data['omadus'], data['temperatuur'], data['rõhk'], data['tihedus_molaarne']))
print(result)
print('\n1-hekseeni PC-SAFT parameetrid:')
print('m: {}'.format(result.x[0]))
print('sigma: {}'.format(result.x[1]))
print('eps_k: {}'.format(result.x[2]))

# kontrolli tulemusi joonisega ----------
x = np.asarray([1])
m = result.x[0]
sigma = result.x[1]
eps_k = result.x[2]
# m = np.asarray([3.05121012]) # kui tahad, võid käsitsi muuta mõnda parameetrit ja näha erinevust
# sigma = np.asarray([3.73067305])
# eps_k = np.asarray([234.04430925])
params_pcsaft = {'m':m, 's':sigma, 'e':eps_k}

npts = 30
t_calc = np.linspace(288, 500, npts) # loome massiivid
vp_calc = np.zeros_like(t_calc)
den_calc = np.zeros_like(t_calc)

# aururõhu joonis
plt.figure()

for i in range(npts):
    vp, x_liq, x_vap = flashTQ(t_calc[i], 0, x, params_pcsaft)
    vp_calc[i] = vp

plt.plot(t_calc, vp_calc)
plt.scatter(data.loc[data['omadus'] == 'aururõhk', 'temperatuur'],
            data.loc[data['omadus'] == 'aururõhk', 'rõhk'])
plt.yscale('log')
plt.title('1-hekseen')
plt.xlabel('Temperatuur (K)')
plt.ylabel('Aururõhk (Pa)')
plt.savefig('1hekseen_vp', dpi=400)
plt.show()

# tiheduse joonis
plt.figure()

for i in range(npts):
    p_den= vp_calc[i] * 1.05 # valime rõhku, mis on natuke kõrgem, kui aururõhk
    den = pcsaft_den(t_calc[i], p_den, x, params_pcsaft, phase='liq')
    den_calc[i] = den

plt.plot(t_calc, den_calc)
plt.scatter(data.loc[data['omadus'] == 'tihedus', 'temperatuur'],
            data.loc[data['omadus'] == 'tihedus', 'tihedus_molaarne'])
plt.title('1-hekseen')
plt.xlabel('Temperatuur (K)')
plt.ylabel('Tihedus (mol/m$^3$)')
plt.savefig('1hekseen_den', dpi=400)
plt.show()
