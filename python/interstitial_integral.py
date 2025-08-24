from scipy.integrate import trapezoid
import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX for all text
plt.rcParams.update({
    "text.usetex": False,              # don't use external LaTeX
    "mathtext.fontset": "cm",          # use Computer Modern
    "font.family": "serif",            # serif text
})

fig1, ax1 = plt.subplots(1)

def planar_action(D, theta):
    if D == 2:
        return 0.5*(1-np.cos(theta)-0.5*np.cos(theta)*(np.sin(theta)**2))
    elif D == 3:
        return (theta-(np.cos(theta)*np.sin(theta))-2*(np.sin(theta)**3*np.cos(theta))/3)/np.pi

def Moss_action(D, sign, theta, r_s):
    if sign == "+":
        if D == 2:
            return 0.5*(1-np.cos(theta)-0.5*np.cos(theta)*(np.sin(theta)**2)) + (3/(16*r_s))*(np.sin(theta))**4
        elif D == 3: 
            return (theta-np.cos(theta)*np.sin(theta)-(2/3)*(np.sin(theta)**3)*np.cos(theta))/np.pi + (4/(5*np.pi))*(np.sin(theta)**5)/r_s
    elif sign == "-": 
        if D == 2:
            return 0.5*(1-np.cos(theta)-0.5*np.cos(theta)*(np.sin(theta)**2)) - (3/(16*r_s))*(np.sin(theta))**4
        elif D == 3: 
            return (theta-np.cos(theta)*np.sin(theta)-(2/3)*(np.sin(theta)**3)*np.cos(theta))/np.pi - (4/(5*np.pi))*(np.sin(theta)**5)/r_s

def Integrand(x, D, E): 
    return np.sqrt(x**(2*D-2) - (E - x**D)**2)

def Gamma(B, D): 
    return ((B/(2*np.pi))**(0.5*D))*np.exp(-B)

theta = [
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
        "value" : np.pi/2, 
        "name": r"$ \frac{\pi}{2}$",
        "linestyle": "solid"
    },
    {
        "value" : np.pi, 
        "name": r"$ \pi$",
        "linestyle": "dotted"
    }
    ]
"""
other values
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
    "value" : np.pi, 
    "name": r"$ \pi$",
    "linestyle": "dotted"
}
"""
D = 3
N = 10000
r_s = np.linspace(0, 100, N)
dx = 0.001

def function():
    for j in range(len(theta)):
        B_trap = np.ones(N)   
        x_norm = np.linspace(0, 1, int(1/dx))
        y_norm = Integrand(x_norm,D,0)
        y_norm_clean = y_norm[~np.isnan(y_norm)]
        B_0 = trapezoid(y_norm_clean, dx = dx)
        
        for i in range(len(r_s)):
            x = np.linspace(r_s[i], r_s[i] + 1, int(1/dx))
            E = (r_s[i]**D - r_s[i]**(D-1)*np.cos(theta[j]["value"]))
            y = Integrand(x, D, E)
            y_clean = y[~np.isnan(y)]
            B_trap[i] = trapezoid(y_clean, dx = dx)
    
        if len(theta) == 1:
            ax1.axhline(planar_action(D, theta[j]["value"]), label="(Planar)", linestyle= "dashed" , color="k")
            ax1.plot(r_s, B_trap/B_0, label= "(Engulfing)", linestyle = theta[j]["linestyle"])
            ax1.plot(r_s, Moss_action(D, "+", theta[j]["value"], r_s), label="(Edge - Convex)", linestyle = "dotted")
            ax1.plot(r_s, Moss_action(D, "-", theta[j]["value"], r_s), label= "(Edge - Concave)", linestyle = "dashdot")
            ax1.legend(loc='upper right', fontsize=15)
        else:
            ax1.plot(r_s, B_trap/B_0, label= r"$ \beta $ = " + theta[j]["name"], linestyle = theta[j]["linestyle"])
            ax1.legend(loc='upper left', fontsize=15)
        
        ax1.set_ylim(0, 2.5)
        ax1.set_ylabel(r"$\frac{B}{B_0}$", fontsize=15, rotation=0, ha="right")
        ax1.set_xlabel(r"$\frac{R_s}{R_0}$", fontsize = 15)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax1.tick_params(axis='both', which='minor', labelsize=15)
        ax1.set_xscale("log")

function()

plt.tight_layout()

if len(theta) == 1:
    plt.savefig("crossover.pdf", bbox_inches="tight")
else: 
    plt.savefig("actionD3.pdf", bbox_inches="tight")
plt.show()