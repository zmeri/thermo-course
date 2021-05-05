'''
Näide aktiivsuse koefitsiendi mudeli arvutusest
'''
import numpy as np


def uniquac(tau, r, q, x):
    '''
    Calculate activity coefficients using the UNIQUAC model

    Parameters
    ----------
    tau : ndarray, shape(n,n)
        Adjustable parameter related to interactions between two different compounds.
        tau[i,i] = 1.
    r : ndarray, shape(n,)
        Pure component parameter related to its molecular van der Waals volume
    q : ndarray, shape(n,)
        Pure component parameter related to its molecular surface area
    x : ndarray, shape(n,)
        Mole fraction of each component in the mixture

    Returns
    -------
    gamma : ndarray, shape(n,)
        Activity coefficient of each component

    References
    ----------
    - D. S. Abrams and J. M. Prausnitz, “Statistical thermodynamics of liquid mixtures: A new expression for the excess Gibbs energy of partly or completely miscible systems,” AIChE Journal, vol. 21, no. 1, pp. 116–128, 1975, doi: 10.1002/aic.690210115.
    '''
    z = 10
    l = z/2 * (r - q) - (r - 1)
    theta = q * x / np.sum(q * x)
    phi = r * x / np.sum(r * x)

    gamma_c = np.log(phi / x) + z/2*q*np.log(theta / phi) + l - phi/x*np.sum(x*l)
    gamma_r = q * (1 - np.log(np.dot(theta, tau)) - np.sum((theta.reshape(-1,1) *
              tau.transpose()) / np.dot(theta, tau).reshape(-1,1), axis=0))

    gamma = gamma_c + gamma_r
    return np.exp(gamma)



# Ülesanne 1 ----------------------------
'''
Arvutada vee ja äädikhappe aktiivsuse koefitsiendid temperatuuril
25 degC. Vee moolprotsent on 80% ja äädikhappe moolprotsent on 20%.
'''
# 0 = vesi, 1 = äädikhappe
t = 298.15 # K
x = np.asarray([0.8, 0.2])
# kõik need parameetrid tulid kirjandusest: T. F. Anderson and J. M. Prausnitz, “Application of the UNIQUAC Equation to Calculation of Multicomponent Phase Equilibria. 1. Vapor-Liquid Equilibria,” Ind. Eng. Chem. Proc. Des. Dev., vol. 17, no. 4, pp. 552–561, Oct. 1978, doi: 10.1021/i260068a028.
r = np.asarray([0.92, 1.90])
q = np.asarray([1.40, 1.80])
a12 = -299.90
a21 = 530.94
tau = np.asarray([[1, np.exp(-a12/t)],
                  [np.exp(-a21/t), 1]])

gamma = uniquac(tau, r, q, x)

print('\n--------  Ülesanne 1  --------')
print('Vee aktiivsuse koefitsient: {}'.format(gamma[0]))
print('Äädikhappe aktiivsuse koefitsient: {}'.format(gamma[1]))
