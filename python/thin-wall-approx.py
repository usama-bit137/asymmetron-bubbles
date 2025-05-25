import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import fsolve

fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)

def V(r, L):
    return 1-(r**2/(r**3-L**3))**2

def V_der(r, L):
    return -((2*r**3/(r**3-L**3)**3))*(r**3+2*L**3)

def Newton_Raphson(x, L, n):
    for i in range(n):
        x += -V(x, L) / V_der(x, L)
        print(x)
    return x

def zero_function(x, L, n):
    def func(x, L): 
        return x**3 - x**2 - L**3

    def func_der(x): 
        return 3*x**2 - 2*x

    for i in range(n):
        x += -func(x, L) / func_der(x)
    return x

# ODE Solution:
def ODE(r, v, t, L):
    return V_der(r, L)

def dS_dt(r, v, L): 
    return ((r**2)*np.sqrt(1+v**2) - r**3 + L**3)

def RK_22(x0, L):
    t_range = 10
    x_range = 10
    dt = 0.000001
    t0 = 0
    v0 = 0  
    
    # Boxes to fill:
    xh_total = []
    t_total = []
    v_total = []

    while t0 < t_range:        
        xh = x0 + dt * v0 / 2
        if x0 < 0:
            break

        if v0 > 0: 
            break
        
        if abs(x0-L) < 0.001: 
            break

        vh = v0 + ODE(x0, v0, t0, L) * dt / 2
        x0 += dt * vh
        v0 += dt * ODE(xh, vh, t0 + dt / 2, L)
        t0 += dt
    
        # Fill the boxes:
        v_total.append(vh)
        xh_total.append(xh)
        t_total.append(t0)
    return np.array([t_total, xh_total, v_total])

N = 1
r_s = np.linspace(2, 5, N)
beta = np.pi
L = (r_s**3-(r_s**2)*np.cos(beta))**(1/3)
S = np.ones(len(r_s))
x = np.linspace(0.001, 30, 1000)

for i in range(len(L)):    
    # solving equations:
    roots = fsolve(lambda x : 1-(x**2/(x**3-L[i]**3))**2, [1, 2*L[i]])
    print(roots)

    r_sol = RK_22(roots[1], L[i])
    ax1.plot(r_sol[1], r_sol[0], label="$\Lambda$ = " + str(np.round(L[i], 2)) + " $R_0$")
    ax2.plot(r_sol[1], V(r_sol[1], L[i]))

    # action integration:
    action = simpson(dS_dt(r_sol[1], r_sol[-1], L[i]), r_sol[0])
    S[i] = action

    # plots:
    ax1.axvline(L[i], linestyle="dashed")
    ax2.axhline(0, lw = "0.5", color = "k")
    ax2.axvline(roots[0], label = "$x_0 = $" + str(round(roots[0], 3)), linestyle = "dashed", color="k")
    ax2.axvline(roots[1], label = "$x_0 = $" + str(round(roots[1], 3)), linestyle = "dashed", color="k")

ax3.plot(L, S/abs(S[0]))
#print("$B_0=$" + str(S[0]))
ax3.set_ylabel(r"$B/|B_0|$", fontsize=15)
ax3.set_xlabel(r"$\Lambda/R_0$", fontsize=15)



ax1.set_ylabel(r"$\tau/R_0$", fontsize=15)
ax1.set_xlabel(r"$r/R_0$", fontsize=15)
ax1.set_title("Euclidean spacetime diagram for bubble wall (closed)")
ax1.legend(loc='upper right', fontsize=15)
ax1.set_xlim(0)
#ax1.set_ylim(0)
ax1.tick_params(axis='both', which='major', labelsize=15)

ax2.legend(loc='lower right', fontsize=15)
ax2.set_ylim(-5, 2.5)
ax2.set_xlim(0, 17)
ax2.set_ylabel("$U(r)$", fontsize=15)
ax2.set_xlabel(r"$r/R_0$", fontsize=15)
plt.show()