import numpy as np
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)

# now we want to show bubble nucleation for a particular value of k.
# the value we are using is pretty good.
def V(chi, k, rho):
    return -0.5*(1-rho)*chi**2 + (chi**4)/4 -(k/3)*(chi**3)

def V_der(chi, k, rho): 
    return -(1-rho)*chi + chi**3 - k*chi**2

# ODE Solution:
def ODE(x, v, t, k, rho):
    return -2*v/t + V_der(x, k, rho)

def RK_22(x0, k, rho):
    
    t_range = 50
    x_range = chi_true+0.01
    dt = 0.001
    t0 = 1e-15
    v0 = 0  
    
    # Boxes to fill:
    xh_total = []
    t_total = []
    v_total = []

    while t0 < t_range:
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

def IntBisec(a_u, a_o, k, rho, N):
    for i in range(N):

        Phi_u = RK_22(a_u, k, rho)
        amid = np.float128(0.5 * (a_u + a_o))

        Phi_mid = RK_22(amid, k, rho)
        
        #testing the tolerance of the solution:
        if abs(Phi_u[0, -1] - Phi_mid[0, -1]) < 0.000001:
            a_u = np.float128(amid)
        else:
            a_o = np.float128(amid)
    return amid

chi = np.linspace(-1.5,2,100)
k, rho = 0.3, 0 # k_max = 0.7
y = V(chi, k, rho)

ax2.plot(chi, y)

chi_false = np.float128((k/2 - np.sqrt(k**2/4 + (1-rho))))
chi_true = np.float128((k/2 + np.sqrt(k**2/4 + (1-rho))))
V_true = V(chi_true, k, rho)
V_false = V(chi_false, k, rho)

ax2.axvline(chi_false, linestyle="dashed", color="k")
ax2.axvline(chi_true, linestyle="dashed", color="k")

ax2.axhline(V_false, linestyle="dashed", color="k")
ax2.axhline(V_true, linestyle="dashed", color="k")
print("hello")
# interval bisection:
a_u = np.float128(chi_true-0.05)
a_o = np.float128(chi_true-0.0000000000000001)

a_mid = IntBisec(a_u, a_o, k, rho, 50)
phi_mid = RK_22(a_mid, k, rho)

t = phi_mid[0]
x = phi_mid[1]
v = phi_mid[-1]

#search algorithm: 
t_cut = t[:30000]
x_cut = x[:30000]
v_cut = v[:30000]

ax1.plot(t_cut, x_cut, color="orange", label="numerical")

ax1.axhline(chi_true, color="black", label="numerical", linestyle="dashed")
ax1.axhline(chi_false, color="black", label="numerical", linestyle="dashed")
ax1.axhline(a_mid, color="red", label="numerical", linestyle="dashed")

#plt.plot(chi,y)
plt.show()