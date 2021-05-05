'''
Näide sellest, kuidas leida olekuvõrrandi interaktsiooni parameetrit andmetest
'''
import numpy as np
import pandas as pd
from pcsaft import pcsaft_p, pcsaft_den, flashTQ, flashPQ # dokumentatsioon: https://pcsaft.readthedocs.io/en/latest/
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def interaction_cost_pcsaft(kij, t, p, x):
    m = np.asarray([3.0576, 2.4653])
    sigma = np.asarray([3.7983, 3.6478])
    eps_k = np.asarray([236.77, 287.35])
    kij_pcsaft = np.asarray([[0, kij],
                            [kij, 0]])

    cost = 0
    for i in range(t.shape[0]):
        param_pcsaft = {'m':m, 's':sigma, 'e':eps_k, 'k_ij': kij_pcsaft}
        try:
            p_fit = flashTQ(t[i], 0, x[i,:], param_pcsaft)[0]
            cost = cost + ((p_fit - p[i]) / p[i] * 100)**2
        except:
            cost = cost + 1e20 # kui ei leia head lahendust, siis lihtsalt lisame suure numbri

    if not np.isfinite(cost):
        cost = 1e60
    return cost


# impordi andmed ---------
data = pd.read_csv('data_hexane_benzene.csv', header=2, delimiter=';')

# kontrolli andmeid joonisega ---------
plt.figure()
plt.scatter(data.loc[data['pressure'] == 101330, 'x1'], data.loc[data['pressure'] == 101330, 'temperature'],
            color='red', marker='o', label='vedelfaas - 101330 Pa')
plt.scatter(data.loc[data['pressure'] == 101330, 'y1'], data.loc[data['pressure'] == 101330, 'temperature'],
            color='red', marker='^', label='gaasifaas - 101330 Pa')
plt.scatter(data.loc[data['pressure'] == 97990, 'x1'], data.loc[data['pressure'] == 97990, 'temperature'],
            color='blue', marker='o', label='vedelfaas - 97990 Pa')
plt.scatter(data.loc[data['pressure'] == 97990, 'y1'], data.loc[data['pressure'] == 97990, 'temperature'],
            color='blue', marker='^', label='gaasifaas - 97990 Pa')
title_text = 'Heksaan + benseen'
plt.xlim([0,1])
plt.title(title_text)
plt.xlabel('Moolosa heksaan')
plt.ylabel('Temperatuur (K)')
plt.legend(frameon=False)
plt.savefig('olekuvõrrandid_andmete_kontroll', dpi=400)
plt.show()


# leia kij parameetrit ---------
bnds = ((0, 0.1),) # Siin paneme piirid paika. Mis piirides parameeter tõenäoliselt on?
x1 = data['x1'].to_numpy().reshape(-1,1)
x = np.hstack((x1, 1-x1))
print('Leian kij lahendajaga. See võib võtta mitu minutit.')
result = differential_evolution(interaction_cost_pcsaft, bounds=bnds,
                                args=(data['temperature'], data['pressure'], x))
print(result)
print('\nHeksaan + benseen kij: {}\n'.format(result.x))

# kontrolli tulemusi joonisega ----------
kij = result.x
# kij = 0.01279835 # kui tahad, võid käsitsi muuta kij ja näha erinevust
m = np.asarray([3.0576, 2.4653])
sigma = np.asarray([3.7983, 3.6478])
eps_k = np.asarray([236.77, 287.35])
kij_pcsaft = np.asarray([[0, kij],
                        [kij, 0]])

plt.figure()

npts = 50
x1 = np.linspace(0, 1, npts)
x1_calc = np.zeros_like(x1)
y1_calc = np.zeros_like(x1)
t_calc = np.zeros_like(x1)
pressures = [101330, 97990]
colors = ['red', 'blue']
first_line = True
for j, p in enumerate(pressures):
    for i in range(npts):
        x = np.asarray([x1[i], 1-x1[i]])
        param_pcsaft = {'m':m, 's':sigma, 'e':eps_k, 'k_ij': kij_pcsaft}
        t_calc[i], x_liq, x_vap = flashPQ(p, 0, x, param_pcsaft)
        x1_calc[i] = x_liq[0]
        y1_calc[i] = x_vap[0]

    if first_line:
        plt.plot(x1_calc, t_calc, color=colors[j], label='PC-SAFT')
        first_line = False
    else:
        plt.plot(x1_calc, t_calc, color=colors[j])
    plt.plot(y1_calc, t_calc, color=colors[j])

plt.scatter(data.loc[data['pressure'] == 101330, 'x1'], data.loc[data['pressure'] == 101330, 'temperature'],
            color='red', marker='o', label='vedelfaas - 101330 Pa')
plt.scatter(data.loc[data['pressure'] == 101330, 'y1'], data.loc[data['pressure'] == 101330, 'temperature'],
            color='red', marker='^', label='gaasifaas - 101330 Pa')
plt.scatter(data.loc[data['pressure'] == 97990, 'x1'], data.loc[data['pressure'] == 97990, 'temperature'],
            color='blue', marker='o', label='vedelfaas - 97990 Pa')
plt.scatter(data.loc[data['pressure'] == 97990, 'y1'], data.loc[data['pressure'] == 97990, 'temperature'],
            color='blue', marker='^', label='gaasifaas - 97990 Pa')
title_text = 'Heksaan + benseen'
plt.title(title_text)
plt.xlim([0,1])
plt.xlabel('Moolosa heksaan')
plt.ylabel('Temperatuur (K)')
plt.legend(frameon=False)
plt.savefig('olekuvõrrandid_kij_leidmine', dpi=400)
plt.show()
