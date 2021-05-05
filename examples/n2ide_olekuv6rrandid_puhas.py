'''
Näide sellest, kuidas olekuvõrrandeid kasutada Pythonis
'''
import numpy as np
import pandas as pd
import CoolProp.CoolProp as cp # dokumentatsioon: http://www.coolprop.org/index.html
from pcsaft import pcsaft_p, pcsaft_den # dokumentatsioon: https://pcsaft.readthedocs.io/en/latest/
import matplotlib.pyplot as plt
from scipy.optimize import minimize

R = 8.314462618 # m^3 Pa K^-1 mol^-1

def soave_redlich_kwong(t, v, c):
    '''
    Calculate pressure using the SRK equation of state

    Parameters
    ----------
    t : float
        Temperature (K)
    v : float
        Molar volume (m^3 mol^-1)
    c : dict
        Fluid constants for the SRK equation of state

    Returns
    -------
    p : float
        Pressure (Pa)
    '''
    a = 0.42748 * R**2 * c['t_crit']**2 / c['p_crit']
    b = 0.08664 * R * c['t_crit'] / c['p_crit']
    alpha = (1 + (0.48 + 1.574 * c['acentric'] - 0.176 * c['acentric']**2) \
            * (1 - (t / c['t_crit'])**0.5))**2
    p = R * t / (v - b) - a * alpha / (v * (v + b))
    return p

def peng_robinson(t, v, c):
    '''
    Calculate pressure using the Peng-Robinson equation of state

    Parameters
    ----------
    t : float
        Temperature (K)
    v : float
        Molar volume (m^3 mol^-1)
    c : dict
        Fluid constants for the Peng-Robinson equation of state

    Returns
    -------
    p : float
        Pressure (Pa)
    '''
    a = 0.45724 * R**2 * c['t_crit']**2 / c['p_crit']
    b = 0.07780 * R * c['t_crit'] / c['p_crit']
    alpha = (1 + (0.37464 + 1.54226 * c['acentric'] - 0.26992 * c['acentric']**2) \
            * (1 - (t / c['t_crit'])**0.5))**2
    p = R * t / (v - b) - a * alpha / (v**2 + 2*b*v - b**2)
    return p


def srk_cost(den, p, t, c):
    ''' Cost function used to solve for the molar density '''
    v = 1/den
    p_calc = soave_redlich_kwong(t, v, c)
    cost = ((p_calc - p) / p * 100)**2
    return cost


def pr_cost(den, p, t, c):
    ''' Cost function used to solve for the molar density '''
    v = 1/den
    p_calc = peng_robinson(t, v, c)
    cost = ((p_calc - p) / p * 100)**2
    return cost


def cubic_den_solver(eos, den_guess, p, t, c):
    '''
    Solves for the molar density using a cubic equation of state

    Parameters
    ----------
    eos : string
        Equation of state to use. Options are soave_redlich_kwong and peng_robinson.
    den_guess: float
        Initial guess for the molar density (mol m^-3)
    p : float
        Pressure (Pa)
    t : float
        Temperature (K)
    c : dict
        Fluid constants for the cubic equation of state

    Returns
    -------
    den : float
        Molar density (mol m^-3)
    '''
    if eos == 'soave_redlich_kwong':
        result = minimize(srk_cost, den_guess, args=(p, t, c))
    elif eos == 'peng_robinson':
        result = minimize(pr_cost, den_guess, args=(p, t, c))
    else:
        raise ValueError('{} is not one of the implemented cubic equations of state'.format(eos))

    den = result.x
    return den


# Lämmastik ------------------
t = np.linspace(130, 373, 20) # K
v = 0.0001 # m^3 mol^-1

param_cubic = {'t_crit':126.192, 'p_crit':3395800.0, 'acentric':0.0372}
p_srk = soave_redlich_kwong(t, v, param_cubic)
p_pr = peng_robinson(t, v, param_cubic)
p_mph = cp.PropsSI('P', 'T', t, 'Dmolar', 1/v, 'NITROGEN')

p_pcsaft = np.zeros_like(t)
x = np.asarray([1.])
m = np.asarray([1.2053])
sigma = np.asarray([3.3130])
eps_k = np.asarray([90.96])
param_pcsaft = {'m':m, 's':sigma, 'e':eps_k}
for i in range(t.shape[0]):
    p_pcsaft[i] = pcsaft_p(t[i], 1/v, x, param_pcsaft)

# kirjuta andmed CSV-failile
output_np = np.hstack((t.reshape(-1,1), p_srk.reshape(-1,1), p_pr.reshape(-1,1),
                    p_mph.reshape(-1,1), p_pcsaft.reshape(-1,1)))
column_names = ['Temperatuur', 'P SRK', 'P PR', 'P HEOS', 'P PC-SAFT']
output = pd.DataFrame(output_np, columns=column_names)
output.to_csv('olekuvõrrandi_tulemused.csv')

# tee joonist
plt.figure()
plt.plot(t, p_srk/100000, label='SRK')
plt.plot(t, p_pr/100000, label='PR')
plt.plot(t, p_mph/100000, label='HEOS')
plt.plot(t, p_pcsaft/100000, label='PC-SAFT')
title_text = 'Lämmastik mahul {}'.format(v) + ' m$^3$ mol$^{-1}$'
plt.title(title_text)
plt.xlabel('Temperatuur (K)')
plt.ylabel('Rõhk (bar)')
plt.legend(frameon=False)
plt.savefig('olekuvõrrandid_nitrogen', dpi=400)
plt.show()


# Metaan ---------------------
t = np.linspace(200, 473, 30) # K
p = 10000000 # Pa

den_mph = cp.PropsSI('Dmolar', 'T', t, 'P', p, 'METHANE')

param_cubic = {'t_crit':190.564, 'p_crit':4599200.0, 'acentric':0.01142}
den_srk = np.zeros_like(t)
den_pr = np.zeros_like(t)
for i in range(t.shape[0]):
    den_srk[i] = cubic_den_solver('soave_redlich_kwong', den_mph[i], p, t[i], param_cubic)
    den_pr[i] = cubic_den_solver('peng_robinson', den_mph[i], p, t[i], param_cubic)

den_pcsaft = np.zeros_like(t)
x = np.asarray([1.])
m = np.asarray([1.])
sigma = np.asarray([3.7039])
eps_k = np.asarray([150.03])
param_pcsaft = {'m':m, 's':sigma, 'e':eps_k}
for i in range(t.shape[0]):
    den_pcsaft[i] = pcsaft_den(t[i], p, x, param_pcsaft, phase='vap')

# tee joonist
plt.figure()
plt.plot(t, den_srk, label='SRK')
plt.plot(t, den_pr, label='PR')
plt.plot(t, den_mph, label='HEOS')
plt.plot(t, den_pcsaft, label='PC-SAFT')
title_text = 'Metaan rõhul {} bar'.format(p/100000)
plt.title(title_text)
plt.xlabel('Temperatuur (K)')
plt.ylabel('Molaarne tihedus (mol m$^{-3}$)')
plt.legend(frameon=False)
plt.savefig('olekuvõrrandid_metaan', dpi=400)
plt.show()


# Atsetoon ----------------------
t = np.linspace(273, 320, 30) # K
p = 100000 # Pa

den_mph = cp.PropsSI('Dmolar', 'T', t, 'P', p, 'ACETONE')

param_cubic = {'t_crit':508.1, 'p_crit':4700000.0, 'acentric':0.3071}
den_srk = np.zeros_like(t)
den_pr = np.zeros_like(t)
for i in range(t.shape[0]):
    den_srk[i] = cubic_den_solver('soave_redlich_kwong', den_mph[i], p, t[i], param_cubic)
    den_pr[i] = cubic_den_solver('peng_robinson', den_mph[i], p, t[i], param_cubic)

den_pcsaft = np.zeros_like(t)
x = np.asarray([1.])
m = np.asarray([2.7448])
sigma = np.asarray([3.2742])
eps_k = np.asarray([232.99])
dipm = np.asarray([2.88])
dip_num = np.asarray([1.])
param_pcsaft = {'m':m, 's':sigma, 'e':eps_k, 'dipm': dipm, 'dip_num': dip_num}
for i in range(t.shape[0]):
    den_pcsaft[i] = pcsaft_den(t[i], p, x, param_pcsaft, phase='liq')

# tee joonist
plt.figure()
plt.plot(t, den_srk, label='SRK')
plt.plot(t, den_pr, label='PR')
plt.plot(t, den_mph, label='HEOS')
plt.plot(t, den_pcsaft, label='PC-SAFT')
title_text = 'Atsetoon rõhul {} bar (vedelfaas)'.format(p/100000)
plt.title(title_text)
plt.xlabel('Temperatuur (K)')
plt.ylabel('Molaarne tihedus (mol m$^{-3}$)')
plt.legend(frameon=False)
plt.savefig('olekuvõrrandid_atsetoon', dpi=400)
plt.show()


# Vesi ----------------------------
t = np.linspace(456, 600, 30) # K
p = 1000000 # Pa

den_mph = cp.PropsSI('Dmolar', 'T', t, 'P', p, 'WATER')

param_cubic = {'t_crit':647.096, 'p_crit':22064000.0, 'acentric':0.3442920843}
den_srk = np.zeros_like(t)
den_pr = np.zeros_like(t)
for i in range(t.shape[0]):
    den_srk[i] = cubic_den_solver('soave_redlich_kwong', den_mph[i], p, t[i], param_cubic)
    den_pr[i] = cubic_den_solver('peng_robinson', den_mph[i], p, t[i], param_cubic)

den_pcsaft = np.zeros_like(t)
x = np.asarray([1.])
m = np.asarray([1.2047])
sigma = np.asarray([0]) # vee sigma ei ole tegelikult 0. Siin paneme lihtsalt suvalist väärtust kuna varsti kasutame valemit, et arvutada sigma sõltuvalt temperatuurist. 
eps_k = np.asarray([353.95])
e_assoc = np.asarray([2425.67])
vol_a = np.asarray([0.0451])
for i in range(t.shape[0]):
    sigma[0] = 3.8395 + 1.2828 * np.exp(-0.0074944*t[i]) - 1.3939 * np.exp(-0.00056029*t[i]) # vee jaoks sigmal on temperatuursõltuvus
    param_pcsaft = {'m':m, 's':sigma, 'e':eps_k, 'e_assoc':e_assoc, 'vol_a':vol_a}
    den_pcsaft[i] = pcsaft_den(t[i], p, x, param_pcsaft, phase='vap')

# tee joonist
plt.figure()
plt.plot(t, den_srk, label='SRK')
plt.plot(t, den_pr, label='PR')
plt.plot(t, den_mph, label='HEOS')
plt.plot(t, den_pcsaft, label='PC-SAFT')
title_text = 'Vesi rõhul {} bar (gaasifaas)'.format(p/100000)
plt.title(title_text)
plt.xlabel('Temperatuur (K)')
plt.ylabel('Molaarne tihedus (mol m$^{-3}$)')
plt.legend(frameon=False)
plt.savefig('olekuvõrrandid_vesi', dpi=400)
plt.show()
