import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# This is the free vacuum case around the seed. No bubbles. We need to find the solutions that interpolate on both of the vacua. 

fig1, (ax1, ax2) = plt.subplots(2,1)
fig2, ax3 = plt.subplots(1)
fig3, ax4 = plt.subplots(1)
fig4, (ax5, ax6) = plt.subplots(2,1)

def V(chi, k, rho):
    return -0.5*(1-rho)*chi**2 + (chi**4)/4 -(k/3)*(chi**3)

def V_der(chi, k, rho): 
    return -(1-rho)*chi + chi**3 - k*chi**2

# ODE Solution:
def ODE(x, v, t, k, rho):
    return -2*v/t + V_der(x, k, rho)/2

# line showing the size of the seed and Compton wavelength.
def RK_22(x0, k, rho, t_s):
    
    t_range = 20
    x_range = chi_true+0.01
    dt = 0.001
    t0 = 1e-15
    v0 = 0  
    
    # Boxes to fill:
    xh_total = []
    t_total = []
    v_total = []

    while t0 < t_range:
        if t0 > t_s:
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

def IntBisec(a_u, a_o, chi, rho, k, N):
    for i in range(N):

        Phi_u = RK_22(a_u, k, rho, t_s)
        amid = np.float128(0.5 * (a_u + a_o))

        Phi_mid = RK_22(amid, k, rho, t_s)
        
        #testing the tolerance of the solution:
        if abs(Phi_u[0, -1] - Phi_mid[0,-1]) < 0.000001:
            a_u = np.float128(amid)
        else:
            a_o = np.float128(amid)
        
        #ax1.plot(Phi_mid[0], Phi_mid[1], color="orange", label="$\chi_{b}(\mu r)$")
    return amid

def sigma_difference(kappa, r_s): 
    return 4*np.pi*(kappa/2)*(2+(2+np.cosh(2*r_s))*np.tanh(r_s)*(1/np.cosh(r_s)**2))*r_s**2

N = 5
t_s = 1.5

sigma_true = np.ones(N)
sigma_false = np.ones(N)
R = np.ones(N)

chi = np.linspace(-1.5,2,100)
k = np.linspace(0, 0.35, N)
rho = 10

for i in range(len(k)):
    y = V(chi, k[i], rho=0)

    ax3.plot(chi, y)

    chi_false = np.float128((k[i]/2 - np.sqrt(k[i]**2/4 + 1)))
    chi_true = np.float128((k[i]/2 + np.sqrt(k[i]**2/4 + 1)))
    V_true = V(chi_true, k[i], rho)

    a_u = 0.1
    a_o = chi_true
    a_mid = IntBisec(a_u, a_o, chi_false, rho, k[i], 50)

    phi_mid = RK_22(a_mid, k[i], rho, t_s)
    
    t = phi_mid[0]
    x = phi_mid[1]
    v = phi_mid[-1]

    #search algorithm:
    max_index = 5000 
    t_cut = t[0:max_index]
    x_cut = x[0:max_index]
    v_cut = v[0:max_index]

    ax1.plot(t_cut, x_cut, label="$\kappa=$" + str(round(k[i], 3))+"$\lambda$")

    #ax1.plot(t_cut[0:500], chi_true/(t_cut[0:500]*np.sqrt(10)*0.5*np.cosh(np.sqrt(10)*0.05))*np.sinh(np.sqrt(10)*t_cut[0:500]), label="analytical", linestyle="dashed", color="red")
    #ax1.plot(t_cut[500:], -np.tanh((t_cut[500:]-R[i])/2)+k/2, linestyle="dashed", color="k")

    # Now we do Simpson's rule on the inside and outside pieces of the energy density

    seed_index = int(t_s*1000)
    ax2.plot(t_cut[0:seed_index], (v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k[i], rho) - V(chi_true, k[i], 0), label="$\kappa=$" + str(round(k[i], 3))+"$\lambda$")
    ax2.plot(t_cut[seed_index:], (v_cut[seed_index:])**2 + V(x_cut[seed_index:], k[i], 0) - V(chi_true, k[i], 0) )

    sigma_true[i] = 4*np.pi*trapezoid(t_cut[0:seed_index], t_cut[0:seed_index]**2*((v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k[i], rho) - V(chi_true, k[i], 0)), dx = 0.001) + trapezoid(t_cut[seed_index:], t_cut[seed_index:]**2*((v_cut[seed_index:])**2 + V(x_cut[seed_index:], k[i], 0) - V(chi_true, k[i], 0)), dx = 0.001)

for i in range(len(k)):
    #y = V(chi, k[i], rho=0)

    #ax3.plot(chi, y)

    chi_false = np.float128((k[i]/2 - np.sqrt(k[i]**2/4 + 1)))
    chi_true = np.float128((k[i]/2 + np.sqrt(k[i]**2/4 + 1)))
    V_true = V(chi_true, k[i], rho)

    a_u = -0.1
    a_o = chi_false
    a_mid = IntBisec(a_u, a_o, chi_false, rho, k[i], 50)

    phi_mid = RK_22(a_mid, k[i], rho, t_s)
    
    t = phi_mid[0]
    x = phi_mid[1]
    v = phi_mid[-1]

    #search algorithm:
    max_index = 5000 
    t_cut = t[0:max_index]
    x_cut = x[0:max_index]
    v_cut = v[0:max_index]

    ax5.plot(t_cut, x_cut, label="$\kappa=$" + str(round(k[i], 3))+"$\lambda$")

    seed_index = int(t_s*1000)
    ax6.plot(t_cut[0:seed_index], (v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k[i], rho) - V(chi_false, k[i], 0), label="$\kappa=$" + str(round(k[i], 3))+"$\lambda$")
    ax6.plot(t_cut[seed_index:], (v_cut[seed_index:])**2 + V(x_cut[seed_index:], k[i], 0) - V(chi_false, k[i], 0))

    sigma_false[i] = 4*np.pi*trapezoid(t_cut[0:seed_index], (((v_cut[0:seed_index])**2 + V(x_cut[0:seed_index], k[i], rho)) - V(chi_false, k[i], 0)), dx = 0.001) + trapezoid(t_cut[seed_index:], ((v_cut[seed_index:])**2 + V(x_cut[seed_index:], k[i], 0) - V(chi_false, k[i], 0)), dx = 0.001)

ax1.axvline(t_s, color="steelblue", label="$r_s = $" + str(t_s) + "$\lambda_0$", linestyle="dashed")
ax1.axvline(1, color="blue", label="$\lambda_0$", linestyle="dotted")
ax1.set_ylabel("$\phi/\phi_0$", fontsize=20)
ax1.set_xlabel("$r/\lambda_0$", fontsize=20)
ax1.legend(loc='lower right', fontsize=15)

ax2.axvline(t_s, color="steelblue", label="$r_s = $" + str(t_s) + "$\lambda_0$", linestyle="dashed")
ax2.axvline(1, color="blue", label="$\lambda_0$", linestyle="dotted")

ax5.axvline(t_s, color="steelblue", label="$r_s = $" + str(t_s) + "$\lambda_0$", linestyle="dashed")
ax5.axvline(1, color="blue", label="$\lambda_0$", linestyle="dotted")
ax6.axvline(t_s, color="steelblue", label="$r_s = $" + str(t_s) + "$\lambda_0$", linestyle="dashed")
ax6.axvline(1, color="blue", label="$\lambda_0$", linestyle="dotted")

ax4.plot(k, sigma_true, linestyle="dashed", color="k", label="$E_{st}$")
ax4.plot(k, sigma_false, color="blue", label="$E_{sf}$")
ax4.plot(k, sigma_true-sigma_false, color="red", label="$\Delta E$")
ax4.plot(k, sigma_difference(k, t_s), color="orange", linestyle="solid", label="$\Delta E$ analytical")

ax4.set_xlabel(r"$\frac{\kappa}{\lambda}$", fontsize=25)
ax4.set_ylabel(r"$\frac{E}{\sigma_w \lambda_0^2}$", fontsize=25, rotation=0, ha="right")
ax4.legend(loc='center right', fontsize=25)
ax4.tick_params(axis='both', which='major', labelsize=15)
ax4.tick_params(axis='both', which='minor', labelsize=15)

ax5.set_ylabel("$\phi/\phi_0$", fontsize=20)
ax5.set_xlabel("$r/\lambda_0$", fontsize=20)
ax5.legend(loc='upper right', fontsize=15)

ax2.set_ylabel("$(\lambda_0/\phi_0^2)ϱ$", fontsize=20)
ax2.set_xlabel("$r/\lambda_0$", fontsize=20)
ax2.legend(loc='upper right', fontsize=15)
ax6.set_ylabel("$(\lambda_0/\phi_0^2)ϱ$", fontsize=20)
ax6.set_xlabel("$r/\lambda_0$", fontsize=20)
ax6.legend(loc='center right', fontsize=15)

ax1.set_xlim(0, max(t_cut))
ax2.set_xlim(0, max(t_cut))
ax2.set_ylim(-0.1)

ax5.set_xlim(0, max(t_cut))
ax6.set_xlim(0, max(t_cut))

ax3.set_ylabel('$\tilde{V}/\lambda\phi_0^4$', fontsize=20)
ax3.set_xlabel('$\phi/\phi_0$', fontsize=20)

ax4.set_xlim(0, 0.35)
ax4.set_xlim(0, 0.35)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()

plt.show()
