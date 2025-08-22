import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

fig1, ax1 = plt.subplots(1)
fig2, ax3 = plt.subplots(1)
fig3, ax4 = plt.subplots(1)
fig4, ax2 = plt.subplots(1)


def V(chi, k, rho):
    return -0.5*(1-rho)*chi**2 + (chi**4)/4 -(k/3)*(chi**3)

def V_der(chi, k, rho): 
    return -(1-rho)*chi + chi**3 - k*chi**2

# ODE Solution:
def ODE(x, v, t, k, rho):
    return -2*v/t + V_der(x, k, rho)/2

# line showing the size of the seed and Compton wavelength.
def RK_22(x0, rho, k, t_s, t_range):
    x_range = chi_true+0.01
    dt = 0.001
    t0 = 1e-15
    v0 = 0  
    
    # Boxes to fill:
    xh_total = []
    t_total = []
    v_total = []
    
    while t0 < t_range:
        if t0 < t_s:
            rho = 10
        else: 
            rho = 0

        xh = x0 + dt * v0 / 2
        if abs(x0) > x_range:
            break

        vh = v0 + ODE(x0, v0, t0, k, rho) * dt / 2
        x0 += dt * vh
        v0 += dt * ODE(xh, vh, t0 + dt / 2, k, rho)
        t0 += dt
    
        # Fill the boxes:
        v_total.append(vh)
        xh_total.append(xh)
        t_total.append(t0)

    return np.array([t_total, xh_total, v_total])

def IntBisec(a_u, a_o, t_s, t_range, k, rho, N):
    for _ in range(N):

        Phi_u = RK_22(a_u, rho, k, t_s, t_range)
        amid = np.float128(0.5 * (a_u + a_o))

        Phi_mid = RK_22(amid, rho, k, t_s, t_range)
        
        #testing the tolerance of the solution:
        if abs(Phi_u[0, -1] - Phi_mid[0, -1]) < 0.000001:
            a_u = np.float128(amid)
        else:
            a_o = np.float128(amid)
        
        #ax1.plot(Phi_mid[0], Phi_mid[1], color="orange", label="$\chi_{b}(\mu r)$")
    return Phi_mid

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
    ax1.plot(t_cut_dw, x_cut_dw, label="$r_s=$" + str(round(t_s[i], 3))+"$\lambda_0$")
    sigma_dw[i] = trapezoid(t_cut_dw[0:seed_index]**2*((v_cut_dw[0:seed_index])**2 + V(x_cut_dw[0:seed_index], k, rho)), dx = dt) + trapezoid(t_cut_dw[seed_index:]**2*((v_cut_dw[seed_index:])**2 + V(x_cut_dw[seed_index:], k, 0)-V(chi_false, k, 0)), dx =dt)

    #ax1.plot(t_cut_vev, x_cut_vev, label="$r_s=$" + str(round(t_s[i], 3))+"$\lambda_0$")
    ax2.plot(t_cut_dw[:seed_index], 0.5*(v_cut_dw[:seed_index])**2 + V(x_cut_dw[:seed_index], k, rho) - (0.5*(v_cut_vev[:seed_index])**2 + V(x_cut_vev[:seed_index], k, rho)), label="$r_s=$" + str(round(t_s[i], 3))+"$\lambda_0$")
    ax2.plot(t_cut_dw[seed_index:], 0.5*(v_cut_dw[seed_index:])**2 + V(x_cut_dw[seed_index:], k, 0) - (0.5*(v_cut_vev[seed_index:])**2 + V(x_cut_vev[seed_index:], k, 0)))

#Decorations:
ax4.plot(t_s, sigma_dw, linestyle="dashed", color="k")
ax4.set_xlabel("$r_s/\lambda_0$")
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
