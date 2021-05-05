'''
Näide faasitasakaalu arvutamisest
'''
import numpy as np
import pandas as pd
from pcsaft import pcsaft_den, pcsaft_fugcoef, flashTQ # dokumentatsioon: https://pcsaft.readthedocs.io/en/latest/
from scipy.optimize import minimize, differential_evolution


# Ülesanne 1 ----------------------------
'''
Arvutada 2-propüülamiini aururõhku temperatuuril 25 degC.
'''
t = 25 + 273.15 # K
q = 0
x = np.asarray([1])
pcsaft_params = {'m': np.asarray([2.5908]),
                 's': np.asarray([3.4777]),
                 'e': np.asarray([231.80]),
                 'vol_a': np.asarray([0.021340]),
                 'e_assoc': np.asarray([932.2])}

pvap, xl, xv = flashTQ(t, q, x, pcsaft_params)

print('\n--------  Ülesanne 1  --------')
print('2-propüülamiini aururõhk: {} Pa'.format(pvap))



# Ülesanne 1 aga lahendajaga --------------------------
def veafunktsioon(p, t, x, params):
    den_l = pcsaft_den(t, p, x, params, phase='liq')
    den_v = pcsaft_den(t, p, x, params, phase='vap')
    fugcoef_l = pcsaft_fugcoef(t, den_l, x, params)
    fugcoef_v = pcsaft_fugcoef(t, den_v, x, params)
    cost = (fugcoef_v - fugcoef_l)**2
    return cost


t = 25 + 273.15 # K
q = 0
x = np.asarray([1])
pcsaft_params = {'m': np.asarray([2.5908]),
                 's': np.asarray([3.4777]),
                 'e': np.asarray([231.80]),
                 'vol_a': np.asarray([0.021340]),
                 'e_assoc': np.asarray([932.2])}

# saame käsitsi proovida paar väärtust, et leida head algväärtust
print('\nLahendajaga lahendamine ----------')
p_test = 60000
cost = veafunktsioon(p_test, t, x, pcsaft_params)
print('Proovitud rõhk= {} Pa'.format(p_test))
print('veafunktsiooni väärtus=', cost)

# lahendamine tuletise meetodiga
p_guess = 50000 # Pa
result = minimize(veafunktsioon, p_guess, args=(t, x, pcsaft_params), method='Nelder-Mead')

# # lahendamine globaalse meetodiga
# bnds = ((50000, 175000),)
# result = differential_evolution(veafunktsioon, bounds=bnds, args=(t, x, pcsaft_params))

pvap = result.x

print('\nlahendaja tulemus:')
print(result)
print('2-propüülamiini aururõhk: {} Pa'.format(pvap))


# Ülesanne 2 ----------------------------
'''
Arvutada vesiniku + n-heksadekaani mullipunkti rõhku temperatuuril 150 degC.
Vesiniku moolprotsent segus on 0.21%.
'''
# 0 = vesinik, 1 = n-heksadekaan
t = 150 + 273.15 # K
q = 0 # moolosa segust, mis läheb gaasifaasi
x0 = 0.0021
x = np.asarray([x0, 1-x0])
pcsaft_params = {'m': np.asarray([1, 6.6485]),
                 's': np.asarray([2.9860, 3.9552]),
                 'e': np.asarray([19.2775, 254.70])}
pcsaft_params['k_ij'] = np.asarray([[0, 0],
                                    [0, 0]])

pvap, xl, xv = flashTQ(t, q, x, pcsaft_params)

print('\n--------  Ülesanne 2  --------')
print('Mullipunkti rõhk: {} Pa'.format(pvap))
