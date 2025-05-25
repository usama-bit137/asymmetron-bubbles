import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

fig1, (ax1, ax2) = plt.subplots(2,1)
fig2, ax3 = plt.subplots(1)
fig3, ax4 = plt.subplots(1)

def V(chi, k, rho):
    return -0.5*(1-rho)*chi**2 + (chi**4)/4 -(k/3)*(chi**3)

def V_der(chi, k, rho): 
    return -(1-rho)*chi + chi**3 - k*chi**2

# ODE Solution:
def ODE(x, v, t, k, rho):
    return -2*v/t + V_der(x, k, rho)/2

# line showing the size of the seed and Compton wavelength.
def RK_22(x0, k, t_s):
    
    t_range = 100
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
            rho=0

        
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

def IntBisec(a_u, a_o, k, N):
    for i in range(N):

        Phi_u = RK_22(a_u, k, t_s)
        amid = np.float128(0.5 * (a_u + a_o))

        Phi_mid = RK_22(amid, k, t_s)
        
        #testing the tolerance of the solution:
        if abs(Phi_u[0, -1] - Phi_mid[0, -1]) < 0.000001:
            a_u = np.float128(amid)
        else:
            a_o = np.float128(amid)
        
        #ax1.plot(Phi_mid[0], Phi_mid[1], color="orange", label="$\chi_{b}(\mu r)$")
    return Phi_mid

N = 10
t_s = 0.5

sigma = np.ones(N)
R = np.ones(N)

chi = np.linspace(-1.5,2,100)
k = np.linspace(0.2, 0.7, N)
rho = 0

for i in range(len(k)):
    y = V(chi, k[i], rho)
    ax3.plot(chi, y)

    chi_false = np.float128((k[i]/2 - np.sqrt(k[i]**2/4 + (1-rho))))
    chi_true = np.float128((k[i]/2 + np.sqrt(k[i]**2/4 + (1-rho))))
    V_true = V(chi_true, k[i], rho)

    # interval bisection:
    a_u = np.float128(chi_true-0.5)
    a_o = np.float128(chi_true-0.000000001)
    phi_mid = IntBisec(a_u, a_o, k[i], 50)

    t = phi_mid[0]
    x = phi_mid[1]
    v = phi_mid[-1]
    
    max_index = 30*1000

    #search algorithm: 
    t_cut = t[0:max_index]
    x_cut = x[0:max_index]
    v_cut = v[0:max_index]
    seed_index = int(t_s*1000)

    ax1.plot(t_cut, x_cut, label="$\kappa=$" + str(round(k[i], 3))+"$\lambda$")
    ax2.plot(t_cut[0:500], 0.5*(v_cut[0:500])**2 + V(x_cut[0:500], k[i], 10) - V(chi_false, k[i], 0), color="red", label="$\kappa=$" + str(round(k[i], 3))+"$\lambda$")
    ax2.plot(t_cut[500:], 0.5*(v_cut[500:])**2 + V(x_cut[500:], k[i], 0) - V(chi_false, k[i], 0), color="red")
    sigma[i] = trapezoid(t_cut[0:max_index], ((v_cut[0:max_index])**2 + V(x_cut[0:max_index], k[i], rho) - V(chi_true, k[i], 0)), dx = 0.001)



#Decorations:
ax1.axvline(t_s, color="steelblue", label="$r_s = $" + str(t_s) + "$\lambda_0$", linestyle="dashed")
ax1.axvline(1, color="blue", label="$\lambda_0$", linestyle="dotted")
ax2.axvline(t_s, color="steelblue", label="$r_s = $" + str(t_s) + "$\lambda_0$", linestyle="dashed")
ax2.axvline(1, color="blue", label="$\lambda_0$", linestyle="dotted")

ax4.plot(k, sigma, linestyle="dashed", color="k")
ax4.set_xlabel("$\kappa/\lambda$")
ax4.set_ylabel("$\sigma$")

# ax1.axhline(chi_true, color="red", label="$\phi_{+}/\phi_0$", linestyle="dashed")
# ax1.axhline(chi_false, color="black", label="$\phi_{-}/\phi_0$", linestyle="dashed")
ax1.set_ylabel("$\phi/\phi_0$", fontsize=20)
ax1.set_xlabel("$r/\lambda_0$", fontsize=20)
ax1.legend(loc='center right', 
           fontsize=15)

ax2.set_ylabel("$(\lambda_0/\phi_0^2)ϱ$", fontsize=20)
ax2.set_xlabel("$r/\lambda_0$", fontsize=20)
ax2.legend(loc='center right', 
           fontsize=15)

ax1.set_xlim(0, 30)
ax2.set_xlim(0, 30)

ax3.set_ylabel('$\tilde{V}/\lambda\phi_0^4$', fontsize=20)
ax3.set_xlabel('$\phi/\phi_0$', fontsize=20)

ax1.grid()
ax2.grid()
ax3.grid()

plt.show()
