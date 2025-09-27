import numpy as np
import math
from config import params

def smooth_activation_function(T_surface, T_abl, smooth_width):
    """
    Smooth activation function to transition ablation effects
    around the ablation temperature T_abl.
    
    Args:
        T_surface: surface temperature
        T_abl: ablation temperature
        smooth_width: width of the smoothing region (K)
        
    Returns:
        activation: value between 0 and 1 indicating ablation activation.
        Fractional values indicate the "mushy zone" where the material is at the phase change but not all parts are ablating.
    """
    if T_surface < T_abl - smooth_width/2:
        return 0.0
    elif T_surface > T_abl + smooth_width/2:
        return 1.0
    else:
        # Smooth transition using a sine function
        return 0.5 * (1 + np.sin(np.pi * (T_surface - T_abl) / smooth_width))


def heat_equation_rhs(t, y):
    """
    Right-hand side function for the heat equation ODE system
    
    Args:
        t: current time
        y: state vector [T0, T1, ..., TN, L]
        params: dictionary containing all physical parameters
        
    Returns:
        dy_dt: derivative vector [dT0/dt, dT1/dt, ..., dTN/dt, dL/dt]
    """
    # Extract temperature and length from state vector
    T = y[:-1]  # Temperature at grid points
    L = y[-1]   # Current leading edge length
    
    # Extract parameters - using consistent naming with problem statement
    N = params['N']
    k = params['k']           # Thermal conductivity [W/(m·K)]
    rho = params['rho']       # Material density [kg/m³]
    cp = params['cp']         # Specific heat [J/(kg·K)]
    r = params['r']           # Leading edge radius [m]
    epsilon = params['epsilon']  # Surface emissivity
    sigma = params['sigma']   # Stefan-Boltzmann constant [W/(m²·K⁴)]
    T_inf = params['T_inf']   # Freestream temperature [K]
    T_r = params['T_r']       # Stagnation temperature [K]
    T_abl = params['T_abl']   # Ablation temperature [K]
    H = params['H']           # Heat of ablation [J/kg]
    qs = params['qs']         # Sutton-Graves heat flux [W/m²]
    smooth_width = params['smooth_width']  # Smoothing width for ablation [K]
    
    h = params.get('h')  # Grid spacing (should be set in params)

    dy_dt = np.zeros_like(y) # Create a zero-vector with dimension matching y.

    # sanity checks
    if len(T) != N + 1:
        raise ValueError(f"y has wrong length: expected {N+1} temperature values, got {len(T)}")
    if N < 2:
        raise ValueError("N must be >= 2 (need at least 3 grid points)")
    if L <= 0:
        raise ValueError("L must be positive")

    # Compute dT/dxi at xi = 1 AND dL/dt (these are coupled)
    # Using Picard iteration with relaxation to solve the coupled system

    max_iter = 10
    tolerance = 1e-8
    omega = 0.5 # Relaxation factor

    dL_dt = 0.0  # Initial guess
    dT_dxi_1 = (T[N] - T[N-1]) / h  # Initial guess for dT/dxi_1 using backward difference

    for i in range(max_iter):
        dT_dxi_1_old = dT_dxi_1 # store values for comparison
        dL_dt_old = dL_dt

        q_in = qs + epsilon * sigma * (T_r**4 - T[N]**4)
        activation = smooth_activation_function(T[N], T_abl, smooth_width)

        # Use BC to evaluate new guess for dL/dt
        dL_dt_new = (activation) * (q_in - (k/L)* dT_dxi_1_old) / (- rho * H)
        # Use BC to evaluate new guess for dT/dxi_1
        dT_dxi_1_new = (q_in + rho * H * dL_dt_new) * (L/k)

        dL_dt = omega * dL_dt_new + (1 - omega) * dL_dt_old
        dT_dxi_1 = omega * dT_dxi_1_new + (1 - omega) * dT_dxi_1_old

        # Check for convergence
        if (abs(dT_dxi_1 - dT_dxi_1_old) < tolerance and 
            abs(dL_dt - dL_dt_old) < tolerance):
            break

    dy_dt[-1] = dL_dt
    
    T_ghost_right = T[N] + h * dT_dxi_1  # Ghost point for right boundary
    d2T_dxi2_1 = (T[N-1] - 2*T[N] + T_ghost_right) / h**2  # Ghost point 2nd partial
    
    dy_dt[N] = (k/(rho*cp)) * (1/L**2) * d2T_dxi2_1 + (1/L)*dT_dxi_1* dL_dt

    # Compute Left Boundary dT/dt
    # Ghost point method for Neumann BC (dT/dξ = 0 at ξ=0)
    T_ghost_left = T[1] # for dT/dξ = 0 at ξ=0, we set T_ghost_left = T[1] 
    d2T_dxi2_0 = (T_ghost_left - 2*T[0] + T[1]) / h**2  
    
    dy_dt[0] = (k/(rho*cp)) * (1/L**2) * d2T_dxi2_0

    # Compute Interior Points
    for i in range(1, N):
        xi_i = i * h
        
        # Second derivative using centered differences
        d2T_dxi2 = (T[i+1] - 2*T[i] + T[i-1]) / h**2
        
        # First derivative for coordinate transformation term
        dT_dxi = (T[i+1] - T[i-1]) / (2*h)
        
        # Coordinate transformation term
        coord_transform = (xi_i / L) * dT_dxi * dL_dt
        
        # Radiation loss term (only for interior points per problem statement)
        radiation_loss = (2/r) * epsilon * sigma * (T[i]**4 - T_inf**4)
        
        # Heat equation
        dy_dt[i] = (k/(rho*cp)) * (1/L**2) * d2T_dxi2 + coord_transform - (1/(rho*cp)) * radiation_loss
    
    return dy_dt
