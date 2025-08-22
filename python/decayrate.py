import numpy as np
import matplotlib.pyplot as plt
from cosmoTransitions.tunneling1D import SingleFieldInstanton  # bounce
from BubbleDet import BubbleConfig, ParticleConfig, BubbleDeterminant

# potential and its derivatives

fig1, ax1 = plt.subplots(1)

def ConstructDet(g, rho):
    msq = (1-rho)

    def V(x):
        return -1 / 2 * msq * x**2 - 1 / 3 * g * x**3 + 1 / 4 * x**4

    def dV(x):
        return -msq * x - g * x**2 +  x**3

    def ddV(x):
        return -msq - 2 * g * x + 3 * x**2

    # minima
    phi_true = (g / 2 + np.sqrt(g**2/4 + msq))
    phi_false =  (g / 2 - np.sqrt(g**2/4 + msq))

    # dimension
    dim = 4

    # nucleation object
    ct_obj = SingleFieldInstanton(
        phi_true,
        phi_false,
        V,
        dV,
        d2V=ddV,
        alpha=(dim - 1),
    )

    # bounce calculation
    profile = ct_obj.findProfile(xtol=1e-9, phitol=1e-9)

    # bounce action
    S0 = ct_obj.findAction(profile)

    # creating bubble config instance
    bub_config = BubbleConfig.fromCosmoTransitions(ct_obj, profile)

    # creating particle instance
    higgs = ParticleConfig(
        W_Phi=ddV,
        spin=0,
        dof_internal=1,
        zero_modes="Higgs",
    )

    # creating bubble determinant instance
    bub_det = BubbleDeterminant(bub_config, higgs)

    # fluctuation determinant
    S1, S1_err = bub_det.findDeterminant()

    """# printing results
    print("S0          :", S0)
    print("S1          :", S1)
    print("err(S1)     :", S1_err)"""

    return [S0, S1, S1_err]

free_things = ConstructDet(0.1, 0)

free_determinant = free_things[1]

def ratio(rho):
     return (ConstructDet(0.1, rho)[1])/free_determinant

def Gamma(x, B):
    if x == 0: 
        prefactor = 1
    else:
        prefactor = ratio(x)

    return (1-x)**3*prefactor*np.exp(B*(1-(1-x)**(3/2)))

x = np.linspace(0, 1, 200)
B = 100

decay_rate = np.zeros(len(x))

for i in range(len(x)):
    decay_rate[i] = Gamma(x[i], B)

ax1.plot(x, decay_rate)
ax1.set_ylabel(r"$\frac{\Gamma(\rho)}{\Gamma(0)}$", fontsize=20)
ax1.set_xlabel(r"$\frac{\rho}{\rho_*}$", fontsize =20)
ax1.set_xlim(0, 1)
ax1.grid("on")
ax1.set_yscale("log")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=15)
plt.show()