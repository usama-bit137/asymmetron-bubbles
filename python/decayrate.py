import numpy as np
import matplotlib.pyplot as plt
from cosmoTransitions.tunneling1D import SingleFieldInstanton  # bounce
from BubbleDet import BubbleConfig, ParticleConfig, BubbleDeterminant

# Use LaTeX for all text
plt.rcParams.update({
    "text.usetex": False,              # don't use external LaTeX
    "mathtext.fontset": "cm",          # use Computer Modern
    "font.family": "serif",            # serif text
})


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
    
    return [S0, S1, S1_err]

g = 2

# calculate the free case
free_things = ConstructDet(g, 0)

print(free_things)

def Ratio(x, g, free_things):
    #pass the vacuum case as an argument here
    S_0 = free_things[0]
    S_1 = free_things[1]

    #calculate the coupled gas actions
    coupled_things = ConstructDet(g, x)
    S_0_coupled = coupled_things[0]
    S_1_coupled = coupled_things[1]
    
    print("Delta S_0 = " + str(S_0_coupled-S_0))
    print("Delta S_1 = " + str(S_1_coupled-S_1))
    print("Delta S_0  + Delta S_1 = "  + str((S_0_coupled-S_0) + (S_1_coupled-S_1)))

    return (S_0_coupled/S_0)**2 * np.exp(-(S_0_coupled-S_0) - (S_1_coupled-S_1))

#x is the gas density.
x = np.linspace(0, 0.9999, 200)
decay_rate = np.zeros(len(x))

for i in range(len(x)):
    decay_rate[i] = Ratio(x[i], g, free_things)
    print(decay_rate[i])
    print("done: " + str(i))

ax1.plot(x, np.log(decay_rate), label="$\Gamma/V$: tree-level + one-loop")
ax1.set_ylabel(r"$\ln\left(\frac{\Gamma(\rho)}{\Gamma(0)}\right)$", fontsize=20)
ax1.set_xlabel(r"$\frac{\rho}{\rho_*}$", fontsize =20)
ax1.set_xlim(0, 1)
ax1.legend(loc='lower right', fontsize=15)
ax1.grid("on")
ax1.set_yscale("log")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=15)
plt.savefig("figure.pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()