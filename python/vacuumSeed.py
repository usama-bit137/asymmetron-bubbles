import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from factory import *

# This is the free vacuum case around the seed. No bubbles. We need to find the solutions that interpolate on both of the vacua. 
fig1, ax1 = plt.subplots(1)
fig2, ax3 = plt.subplots(1)
fig3, ax4 = plt.subplots(1)
fig4, ax5 = plt.subplots(1)
fig5, (ax2, ax6) = plt.subplots(2, 1)
fig6, ax7 = plt.subplots(1)

# Use LaTeX for all text
plt.rcParams.update({
    "text.usetex": False,              # don't use external LaTeX
    "mathtext.fontset": "cm",          # use Computer Modern
    "font.family": "serif",            # serif text
})


# ODE Solution:

def mu(kappa, chi): 
    return np.sqrt(1 + (kappa/2)**2  + np.sign(chi)*(kappa/2)*np.sqrt(1+kappa**2/4))

def sigma(chi, rho, kappa): 
    return (rho/(2*np.sqrt(2)))*chi**2*mu(kappa, chi)/((mu(kappa, chi)+np.sqrt(rho/2))**2)

N = 50
def sampler(x): 
    # linspace sampler
    return x**2

t_s = sampler(np.linspace(0, np.sqrt(5), N))

sigma_true = np.ones(N)
sigma_false = np.ones(N)
R = np.ones(N)

chi = np.linspace(-1.5,2,100)
k = 0.01
rho = 10

chi_false = np.float128((k/2 - np.sqrt(k**2/4 + 1)))
chi_true = np.float128((k/2 + np.sqrt(k**2/4 + 1)))

a_u = 0
a_o = chi_true
    
for i in range(len(t_s)):
    if(t_s[i] < 30): 
        phi_mid = IntBisec(a_u, a_o, chi_true, t_s[i], rho, k, 100)
    else: 
        phi_mid = IntBisec(a_u, a_o, chi_true, t_s[i], rho, k, 200)

    t = phi_mid[0]
    x = phi_mid[1]
    v = phi_mid[-1]

    #search algorithm:
    max_index = int(t_s[i] + 10)*1000 
    t_cut = t[0:max_index]
    x_cut = x[0:max_index]
    v_cut = v[0:max_index]
    
    seed_index = int(t_s[i]*1000)

    ax1.plot(t, x, label="$R_s=$" + str(round(t_s[i], 3))+ "$\lambda_0$")
    ax2.plot(t_cut[0:seed_index], (v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k, rho), label="$\kappa=$" + str(round(k, 3))+"$\lambda$")
    ax2.plot(t_cut[seed_index:], (v_cut[seed_index:])**2 + V(x_cut[seed_index:], k, 0) - V(chi_true, k, 0))

    dt = abs(t_cut[1]-t_cut[0])
     
    sigma_true[i] = trapezoid(0.5*(v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k, rho) , dx = dt) + trapezoid((v_cut[seed_index:])**2 + V(x_cut[seed_index:], k, 0) - V(chi_true, k, 0), dx = dt)

a_o = chi_false

for i in range(len(t_s)):
    
    if(t_s[i] < 30): 
        phi_mid = IntBisec(a_u, a_o, chi_true, t_s[i], rho, k, 100)
    else: 
        phi_mid = IntBisec(a_u, a_o, chi_true, t_s[i], rho, k, 200)

    t = phi_mid[0]
    x = phi_mid[1]
    v = phi_mid[-1]

    #search algorithm:
    max_index = int(t_s[i] + 10)*1000
    t_cut = t[0:max_index]
    x_cut = x[0:max_index]
    v_cut = v[0:max_index]

    ax5.plot(t, x, label="$R_s=$" + str(round(t_s[i], 3))+ "$\lambda_0$")
    
    seed_index = int(t_s[i]*1000)
    ax6.plot(t_cut[0:seed_index], (v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k, rho),  
             label="$\kappa=$" + str(round(k, 3)) + "$\lambda$")
    
    ax6.plot(t_cut[seed_index:], (v_cut[seed_index:])**2 + V(x_cut[seed_index:], k, 0) - V(chi_false, k, 0))
    
    dt = abs(t_cut[1]-t_cut[0])
    sigma_false[i] = trapezoid(0.5*(v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k, rho), dx = dt) + trapezoid(((v_cut[seed_index:])**2 + V(x_cut[seed_index:], k, 0) - V(chi_false, k, 0)), dx = dt)
    #ax7.plot(t_cut[0:seed_index], V_der_2(x_cut[0:seed_index], k, rho))
    ax7.plot(t_cut[seed_index:], V_der_2(x_cut[seed_index:], k, 0),label="$R_s=$" + str(round(t_s[i], 3))+ "$\lambda_0$")

ax7.set_xlabel("$r/\lambda_0$")
ax7.set_ylabel(r"$V_\text{eff}''(\phi)$")
ax7.legend(loc='lower right', fontsize=15)

#ax4.axhline(sigma(chi_true, rho, k) - sigma(chi_false, rho, k) , linestyle="dashed", color="k", label="analytic asymptotics")
ax4.plot(t_s, (sigma_true-sigma_false), color="red", label="$[\sigma]$")
ax4.axhline(15*k/4, color="red", label="$[\sigma]$")
ax4.set_xlabel(r"$\frac{R_s}{\lambda_0}$", fontsize=25)
ax4.set_ylabel(r"$\frac{\sigma}{\sigma_w}$", fontsize=25, rotation=0, ha="right")
ax4.legend(loc='lower right', fontsize=15)
ax4.tick_params(axis='both', which='major', labelsize=15)
ax4.tick_params(axis='both', which='minor', labelsize=15)
#ax4.axhline(3*np.sqrt(2)*k/4, linestyle="dotted", color="purple")

ax1.set_ylabel("$\phi/\phi_0$", fontsize=20)
ax1.set_xlabel("$r/\lambda_0$", fontsize=20)
ax1.legend(loc='center right', fontsize=15)

ax5.set_ylabel("$\phi/\phi_0$", fontsize=20)
ax5.set_xlabel("$r/\lambda_0$", fontsize=20)
ax5.legend(loc='center right', fontsize=15)

ax2.set_ylabel("$(\lambda_0/\phi_0^2)ϱ$", fontsize=20)
ax2.set_xlabel("$r/\lambda_0$", fontsize=20)
ax2.legend(loc='center right', fontsize=15)

ax6.set_ylabel("$(\lambda_0/\phi_0^2)ϱ$", fontsize=20)
ax6.set_xlabel("$r/\lambda_0$", fontsize=20)
ax6.legend(loc='center right', fontsize=15)

ax1.set_xlim(0, max(t_cut))
ax2.set_xlim(0, max(t_cut))

ax5.set_xlim(0, max(t_cut))
ax6.set_xlim(0, max(t_cut))

ax3.set_ylabel('$\tilde{V}/\lambda\phi_0^4$', fontsize=20)
ax3.set_xlabel('$\phi/\phi_0$', fontsize=20)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()

plt.show()