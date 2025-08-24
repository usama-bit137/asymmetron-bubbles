import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from factory import *
fig1, ax1 = plt.subplots(1)
fig2, ax3 = plt.subplots(1)
fig3, ax4 = plt.subplots(1)
fig4, ax2 = plt.subplots(1)

N = 10
t_s = np.linspace(0, 5, N)
sigma_dw = np.ones(N)
sigma_vev = np.ones(N)

chi = np.linspace(-1.5,2,100)
k = 0.2
t_range_dw = 100
rho = 10

for i in range(len(t_s)):
    t_range_vev = t_s[i] + 50
    chi_false = np.float128((k/2 - np.sqrt(k**2/4 + 1)))
    chi_true = np.float128((k/2 + np.sqrt(k**2/4 + 1)))
    y = V(chi, k, 0) - V(chi_false, k, 0)
    ax3.plot(chi, y)

    V_true = V(chi_true, k, 0)

    # interval bisection:
    a_u = 0
    a_o_dw = chi_true
    a_o_vev = chi_false

    phi_mid_dw = IntBisec(a_u, a_o_dw, t_s[i], t_range_dw, k, rho, 50)
    phi_mid_vev = IntBisec(a_u, a_o_vev, t_s[i], t_range_vev, k, rho, 200)

    t_dw = phi_mid_dw[0]
    x_dw = phi_mid_dw[1]
    v_dw = phi_mid_dw[-1]

    t_vev = phi_mid_vev[0]
    x_vev = phi_mid_vev[1]
    v_vev = phi_mid_vev[-1]
    
    # Array slicing:
    max_index = 30*1000
    t_cut_dw = t_dw[0:max_index]
    x_cut_dw = x_dw[0:max_index]
    v_cut_dw = v_dw[0:max_index]

    t_cut_vev = t_vev[0:max_index]
    x_cut_vev = x_vev[0:max_index]
    v_cut_vev = v_vev[0:max_index]

    seed_index = int(t_s[i]*1000)
    dt = 0.001
    ax1.plot(t_cut_dw, x_cut_dw, label="$R_s=$" + str(round(t_s[i], 3))+"$\lambda_0$")
    sigma_dw[i] = trapezoid(t_cut_dw[0:seed_index]**2*((v_cut_dw[0:seed_index])**2 + V(x_cut_dw[0:seed_index], k, rho)), dx = dt) + trapezoid(t_cut_dw[seed_index:]**2*((v_cut_dw[seed_index:])**2 + V(x_cut_dw[seed_index:], k, 0)-V(chi_false, k, 0)), dx =dt)

    #ax1.plot(t_cut_vev, x_cut_vev, label="$R_s=$" + str(round(t_s[i], 3))+"$\lambda_0$")
    ax2.plot(t_cut_dw[:seed_index], 0.5*(v_cut_dw[:seed_index])**2 + V(x_cut_dw[:seed_index], k, rho) - (0.5*(v_cut_vev[:seed_index])**2 + V(x_cut_vev[:seed_index], k, rho)), label="$R_s=$" + str(round(t_s[i], 3))+"$\lambda_0$")
    ax2.plot(t_cut_dw[seed_index:], 0.5*(v_cut_dw[seed_index:])**2 + V(x_cut_dw[seed_index:], k, 0) - (0.5*(v_cut_vev[seed_index:])**2 + V(x_cut_vev[seed_index:], k, 0)))

#Decorations:
ax4.plot(t_s, sigma_dw, linestyle="dashed", color="k")
ax4.set_xlabel("$R_s/\lambda_0$")
ax4.set_ylabel("$E$")

ax1.set_ylabel("$\phi/\phi_0$", fontsize=20)
ax1.set_xlabel("$r/\lambda_0$", fontsize=20)
ax1.legend(loc='center right', fontsize=15)

ax2.set_ylabel("$(\lambda_0/\phi_0^2)ϱ$", fontsize=20)
ax2.set_xlabel("$r/\lambda_0$", fontsize=20)
ax2.legend(loc='center right', fontsize=15)

ax1.set_xlim(0, 30)
ax2.set_xlim(0, 30)

ax3.set_ylabel('$\tilde{V}/\lambda\phi_0^4$', fontsize=20)
ax3.set_xlabel('$\phi/\phi_0$', fontsize=20)

ax1.grid()
ax2.grid()
ax3.grid()

plt.show()
