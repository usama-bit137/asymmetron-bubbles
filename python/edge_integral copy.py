from scipy.integrate import trapezoid
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)

def alpha(x, beta, r_s): 
    return np.arctan(x*np.sin(beta)/(r_s-x*np.cos(beta)))

def Integrand(x, E): 
    return np.sqrt(x**4 - (x**3*(2-3*np.cos(alpha(x, beta, r_s))+(np.cos(alpha(x, beta, r_s)))**3)/3 - E)**2)

def Gamma(B, D): 
    return ((B/(2*np.pi))**(0.5*(D+1)))*np.exp(-B)

beta = [
        {
            "value" : np.pi/8, 
            "name": r"$ \frac{\pi}{8}$",
            "linestyle": "dashed"
        }, 
        {
            "value" : np.pi/4, 
            "name": r"$ \frac{\pi}{4}$",
            "linestyle": "dashdot"
        },
        {
            "value" : np.pi/2 -0.5, 
            "name": r"$ \frac{\pi}{2}$- 0.5",
            "linestyle": "solid"
        },
        {
            "value" : np.pi/2, 
            "name": r"$ \frac{\pi}{2}$",
            "linestyle": "solid"
        },
        {
            "value" : np.pi/2 + 0.5, 
            "name": r"$ \frac{\pi}{2}$+ 0.5",
            "linestyle": "solid"
        },
        {
            "value" : np.pi, 
            "name": r"$ \pi$",
            "linestyle": "dotted"
        }
        ]

D = 3
N = 10000
r_s = np.linspace(0, 100, N)
dx = 0.001

for j in range(len(beta)):
    B_trap = np.ones(N)   
    x_norm = np.linspace(0, 1, int(1/dx))
    y_norm = Integrand(x_norm, 0)
    y_norm_clean = y_norm[~np.isnan(y_norm)]
    B_0 = trapezoid(y_norm_clean, dx = dx)
    
    ax1.plot(x_norm, y_norm)

    for i in range(len(r_s)):
        x = np.linspace(r_s[i], r_s[i] + 1, int(1/dx))
        E = (r_s[i]**3 - r_s[i]**(2)*np.cos(beta[j]["value"]))
        y = Integrand(x, E)
        y_clean = y[~np.isnan(y)]
        B_trap[i] = trapezoid(y_clean, dx = dx)
    
    ax2.plot(r_s, B_trap/B_0, label= r"$ \beta $ = " + beta[j]["name"], linestyle = beta[j]["linestyle"])
    ax3.plot(r_s, Gamma(B_trap/B_0, 0), label= r"$ \beta $ = " + beta[j]["name"], linestyle = beta[j]["linestyle"])

ax2.set_ylim(0, 3)
ax1.set_title("The $D=$" + str(D) + " bubble action integrand", fontsize = 20)
ax1.set_ylabel(r"$\frac{d\mathcal{B}}{dr}$", fontsize=20, rotation=0, ha="right")
ax1.set_xlabel(r"$\frac{r}{\mathcal{R}_0}$", fontsize = 20)

ax2.legend(loc='upper left', fontsize=15)
ax2.set_ylabel(r"$\frac{\mathcal{B}}{\mathcal{B}_0}$", fontsize=25, rotation=0, ha="right")
ax2.set_xlabel(r"$\frac{r_s}{\mathcal{R}_0}$", fontsize = 25)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='minor', labelsize=15)
ax2.set_xscale("log")


ax3.legend(loc='lower left', fontsize=15)
ax3.set_ylabel("$\Gamma$", fontsize=25, rotation=0, ha="right")
ax3.set_xlabel(r"$\frac{r_s}{\mathcal{R}_0}$", fontsize = 25)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.tick_params(axis='both', which='minor', labelsize=15)
ax3.set_xscale("log")

plt.show()