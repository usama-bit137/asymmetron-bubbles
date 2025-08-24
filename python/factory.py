import numpy as np

def ODE(x, v, t, k, rho):
    return -2*v/t + V_der(x, k, rho)/2

def RK_22(x0, chi, k, rho, t_s):
    
    t_range = t_s + 50
    x_range = abs(chi)+0.01
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

def IntBisec(a_u, a_o, chi, t_s, rho, k, N):
    for _ in range(N):
        Phi_u = RK_22(a_u, chi, k, rho, t_s)
        amid = np.float128(0.5 * (a_u + a_o))
        Phi_mid = RK_22(amid, chi, k, rho, t_s)
        
        #testing the tolerance of the looping:

        if abs(Phi_u[0, -1] - Phi_mid[0,-1]) < 0.000001:
            a_u = np.float128(amid)
        else:
            a_o = np.float128(amid)
        
    return Phi_mid

def V(chi, k, rho):
    return -0.5*(1-rho)*chi**2 + (chi**4)/4 -(k/3)*(chi**3)

def V_der(chi, k, rho): 
    return -(1-rho)*chi + chi**3 - k*chi**2

def V_der_2(chi, k, rho): 
    return -(1-rho) + 3*chi**2 - 2*k*chi

