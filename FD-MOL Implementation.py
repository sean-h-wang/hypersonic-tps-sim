# Comparative Analysis of **Method of Lines**, Spectral, and Discontinuous Galerkin Methods for Laptop-Friendly Hypersonic TPS Radiation-Ablation Modeling

# Last edit: 9/24/2025

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # or use scipy.integrate.solve_ivp
import webbrowser
import matplotlib.animation as animation
import os
import sys
import matplotlib
matplotlib.use('TkAgg')


# ==============================================================================
# HYPERSONIC LEADING EDGE HEAT TRANSFER SIMULATOR - FD-MOL VERSION
# This program simulates heat conduction with ablation in hypersonic vehicle 
# leading edges using Finite Difference Method of Lines (FD-MOL)
# ==============================================================================

# ==============================================================================
# CONSOLIDATED PARAMETERS DICTIONARY
# ==============================================================================

# Materials database with properties
MATERIALS = {
    '1': {'name': 'Carbon-Carbon',      'k': 80.0,  'rho': 1600.0,  'cp': 1050.0,   'T_abl': 2200, 'H': 5.0e6,  'epsilon': 0.73},
    '2': {'name': 'HfC UHTC',           'k': 25.0,  'rho': 12600.0, 'cp': 280.0,    'T_abl': 4200, 'H': 8.0e6,  'epsilon': 0.60},
    '3': {'name': 'TaC UHTC',           'k': 22.0,  'rho': 14500.0, 'cp': 265.0,    'T_abl': 4100, 'H': 7.5e6,  'epsilon': 0.65},
    '4': {'name': 'ZrC UHTC',           'k': 20.0,  'rho': 6600.0,  'cp': 300.0,    'T_abl': 3700, 'H': 7.2e6,  'epsilon': 0.70},
    '5': {'name': 'Pyrolytic Graphite', 'k': 300.0, 'rho': 2200.0,  'cp': 1000.0,   'T_abl': 3000, 'H': 6.8e6,  'epsilon': 0.60},
    '6': {'name': 'AVCOAT Ablative',    'k': 0.5,   'rho': 500.0,   'cp': 1500.0,   'T_abl': 1200, 'H': 2.5e6,  'epsilon': 0.85},
    '7': {'name': 'Iridium Alloy',      'k': 147.0, 'rho': 22300.0, 'cp': 130.0,    'T_abl': 2600, 'H': 4.8e6,  'epsilon': 0.30},
}


# Material properties obtained through literature review and AI-assisted compilation
# from scattered academic and industry sources due to limited availability of 
# comprehensive hypersonic material databases

params = {

    # Material properties (from problem statement)
    'material_name': None,        # Material name (for display)
    'k': None,                    # Thermal conductivity [W/(m·K)]
    'rho': None,                  # Material density [kg/m³]
    'cp': None,                   # Specific heat [J/(kg·K)]
    'T_abl': None,                # Ablation temperature [K]
    'H': None,                    # Heat of ablation [J/kg]
    'epsilon': None,              # Surface emissivity
        
    # Geometric properties (from problem statement)
    'r': 0.001,                   # Leading edge radius [m]
    'L0': 0.10,                   # Initial leading edge length [m]

    # Initial Conditions (from problem statement)
    'T0': 300.0,                  # Initial temperature [K] (uniform)
    
    # Environmental conditions (calculated - initially None)
    'altitude': None,             # Altitude [m] (user input)
    'M_inf': None,                # Mach number (user input)
    'T_inf': None,                # Freestream temperature [K] (calculated)
    'P_inf': None,                # Freestream pressure [Pa] (calculated)
    'rho_inf': None,              # Freestream density [kg/m³] (calculated)
    'T_r': None,                  # Stagnation temperature [K] (calculated)
    'V_inf': None,                # Flight velocity [m/s] (calculated)
    
    # Physical constants (global constants)
    'gamma': 1.4,                 # Ratio of specific heats for air
    'R_gas': 287.052,             # Gas constant for air [J/(kg·K)]
    'sigma': 5.67e-8,             # Stefan-Boltzmann constant [W/(m²·K⁴)]
    'C_sg': 1.74153e-4,           # Sutton-Graves constant [kg^0.5/(m^0.5·s^3)]
    
    # Heat flux (calculated - initially None)
    'qs': None,                   # Sutton-Graves heat flux [W/m²]
    
    # Numerical simulation parameters
    'N': 100,                     # Number of spatial grid points - 1 (N intervals, N+1 points)
    'smooth_width': 10.0,         # Width of smoothing region for ablation activation [K]
    'h': None,                    # Grid spacing (calculated by setup_grid)
    'dt': 0.001,                  # Time step for manual RK4 (if used) [s]
    't_max' : None,               # Maximum simulation time [s] (user input)
    'strategy': 'Radau'           # manual RK4' or 'auto RK4' or 'Radau'
}



def get_atmospheric_properties(altitude_m: float):
    """
    Simplified U.S. Standard Atmosphere (NASA Glenn).
    Valid up to ~25 km. Returns temperature [K], pressure [Pa],
    and density [kg/m³].
    """
    h = altitude_m

    if h < 11000:  # Troposphere
        T = 288.15 - 0.00649 * h
        P = 101325.0 * (T / 288.15) ** 5.256
    elif h < 25000:  # Lower stratosphere
        T = 216.65
        P = 22632.06 * math.exp(-0.000157 * (h - 11000))
    else:  # Upper stratosphere (approx.)
        T = 141.94 + 0.00299 * h
        P = 2488.0 * (T / 216.65) ** (-11.388)

    R = params['R_gas']
    rho = P / (R * T)
    return T, P, rho


def print_material_selection():
    """Display material selection menu."""
    print("\nMATERIAL SELECTION:")
    for i in range(1, len(MATERIALS) + 1):
        mat = MATERIALS[str(i)]
        print(f"{i}. {mat['name']} (k={mat['k']} W/m·K, rho={mat['rho']} kg/m³, cp={mat['cp']} J/kg·K, T_abl={mat['T_abl']} K)")


def update_material_properties(material_choice: str):
    """Update params dictionary with selected material properties."""
    if material_choice in MATERIALS:
        material = MATERIALS[material_choice]
        params.update({
            'material_name': material['name'],
            'k': material['k'],
            'rho': material['rho'],
            'cp': material['cp'],
            'T_abl': material['T_abl'],
            'H': material['H'],
            'epsilon': material['epsilon']
        })
    else:
        print(f"Warning: Material choice '{material_choice}' not found.")


def calculate_sutton_graves_heating(M_inf, altitude, R_n):
    """
    Calculate hypersonic heat flux using simplified Sutton-Graves equation
    q"_s = C * sqrt(rho_inf/R_n) * V_inf^3
    
    Args:
        M_inf: Mach number
        altitude: Altitude [m]
        R_n: Nose radius [m]
    
    Returns:
        qs: Sutton-Graves heat flux [W/m²]
        V_inf: Flight velocity [m/s]
    """
    # Get atmospheric properties
    T_inf, _, rho_inf = get_atmospheric_properties(altitude)
    
    # Air properties
    gamma = params['gamma']
    R_gas = params['R_gas']
    
    # Calculate flight velocity
    c_inf = math.sqrt(gamma * R_gas * T_inf)  # Speed of sound
    V_inf = M_inf * c_inf                   # Flight velocity
    
    # Sutton-Graves equation - simplified form
    C_sg = params['C_sg']
    qs = C_sg * math.sqrt(rho_inf / R_n) * V_inf**3
    
    return qs, V_inf


def calculate_stagnation_temperature(T_inf, M_inf, gamma=1.4):
    """
    Calculate stagnation temperature using Mach number relation:
    T_r = T_inf * (1 + (gamma-1)/2 * M_inf^2)
    """
    T_r = T_inf * (1 + (gamma - 1) / 2 * M_inf**2)
    return T_r


def update_calculated_parameters():
    """Update params dictionary with calculated atmospheric and heat flux parameters."""
    # Get atmospheric properties
    T_inf, P_inf, rho_inf = get_atmospheric_properties(params['altitude'])
    
    # Calculate heat flux and flight velocity
    qs, V_inf = calculate_sutton_graves_heating(params['M_inf'], params['altitude'], params['r'])
    
    # Calculate stagnation temperature
    T_r = calculate_stagnation_temperature(T_inf, params['M_inf'], params['gamma'])
    
    # Update params dictionary
    params.update({
        'T_inf': T_inf,
        'P_inf': P_inf,
        'rho_inf': rho_inf,
        'T_r': T_r,
        'V_inf': V_inf,
        'qs': qs,
        'h' : 1/params['N']
    })


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


def solve_heat_equation_fd_mol(strategy: str):
    """
    Solve the heat equation using Finite Difference Method of Lines

    Args: 
        strategy: 'manual RK4' or 'auto RK4' or 'Radau' determines solve strategy
    
    Returns:
        t_sol: time array
        T_sol: temperature solution array (time x space)
        L_sol: leading edge length array
    """
    

    # Setup outputs:
    t_sol = []
    T_sol = []
    L_sol = []

    # Retrieve Variables
    T0 = params['T0']
    L0 = params['L0']
    N = params['N']
    t_max = params['t_max']

    # Initial condition
    y0 = np.concatenate([T0 * np.ones(N + 1), [L0]])   # Combined initial state
    tspan = [0, t_max]

    def ablation_complete(t, y):
        return y[-1] - L0/10  # L - L0/10 = 0

    ablation_complete.terminal = True
    ablation_complete.direction = -1  # Trigger when decreasing

    if strategy == 'manual RK4':
        t_current = tspan[0]
        y_current = y0.copy()
        dt = params['dt']

        t_history = [t_current]
        y_history = [y_current.copy()]

        while t_current < tspan[1]:
            L_current = y_current[-1]
            if L_current < params['L0'] / 10:
                break
                
            # Perform RK4 steps
            k1 = heat_equation_rhs(t_current, y_current)
            k2 = heat_equation_rhs(t_current + dt/2, y_current + dt/2 * k1)
            k3 = heat_equation_rhs(t_current + dt/2, y_current + dt/2 * k2)
            k4 = heat_equation_rhs(t_current + dt, y_current + dt * k3)
            
            # RK4 update
            y_current = y_current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t_current += dt
            
            # Store results
            t_history.append(t_current)
            y_history.append(y_current.copy())

        # Convert to output format
        y_array = np.array(y_history)
        t_sol = np.array(t_history)
        T_sol = y_array[:, :-1]
        L_sol = y_array[:, -1]

    elif strategy == 'auto RK4':

        try:
            sol = solve_ivp(heat_equation_rhs, tspan, y0, method='RK45', events=ablation_complete, dense_output=True, rtol=1e-8, atol=1e-10)

            if sol.success and len(sol.t) > 0:
                t_sol = sol.t
                y_sol = sol.y.T

                T_sol = y_sol[:, :-1]
                L_sol = y_sol[:, -1]
            else:
                print(f"Auto RK4 failed: {sol.message}")
                return None, None, None
            
        except Exception as e:
            print(f"Auto RK4 encountered an error: {e}")
            return None, None, None

    elif strategy == 'Radau':

        try:
            sol = solve_ivp(heat_equation_rhs, tspan, y0, method='Radau', events=ablation_complete, dense_output=True, rtol=1e-8, atol=1e-10)

            if sol.success and len(sol.t) > 0:
                t_sol = sol.t
                y_sol = sol.y.T

                T_sol = y_sol[:, :-1]
                L_sol = y_sol[:, -1]
            else:
                print(f"Radau failed: {sol.message}")
                return None, None, None
            
        except Exception as e:
            print(f"Radau encountered an error: {e}")
            return None, None, None

    else:
        print(f"Error: Unknown strategy '{strategy}'")
        return None, None, None
        
    return t_sol, T_sol, L_sol


def visualize_results(t_sol, T_sol, L_sol):
    """
    Conference-ready shell for hypersonic leading edge visualization.
    2x3 layout with rectangular subplots.
    Simple animation controls: Start/Stop and Reset buttons.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from matplotlib.widgets import Button
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import interp1d

    # Create spatial grid for plotting
    xi = np.linspace(0, 1, params['N'] + 1)  # Normalized grid

    # Create figure and axes
    fig, axes = plt.subplots(2, 3)  # wide and tall for projector
    
    
    # Maximize AFTER the figure is fully created and rendered
    plt.show(block=False)  # Show window first
    
    # Now maximize
    manager = fig.canvas.manager  
    root = manager.window.winfo_toplevel()
    root.update_idletasks()  # Let it fully render first
    root.state('zoomed')     # Then maximize
    
    print("Opening maximized window...")

    plt.subplots_adjust(wspace=0.45, hspace=0.5, bottom=0.12)  # More spacing, room for controls

    # --- Add big figure title
    fig.suptitle("Hypersonic Leading Edge Heat Transfer Simulation", fontsize=20, y=0.97)
    fig.text(0.5, 0.91, f"Material: {params['material_name']}, Mach {params['M_inf']}, Altitude: {int(params['altitude'])} m, Strategy: {params['strategy']}",
        ha='center', fontsize=14)

    # --- Top row subplots
    axes[0,0].set_title("Leading Edge Length vs Time", fontsize=12)
    axes[0,0].set_xlabel("Time [s]", fontsize=10)
    axes[0,0].set_ylabel("L [m]", fontsize=10)
    axes[0,0].grid(True)
    axes[0,0].set_box_aspect(0.6)  # rectangle shape
    axes[0,0].plot(t_sol, L_sol, color='blue', linewidth=1)
    axes[0,0].set_xlim(0, np.max(t_sol))
    axes[0,0].set_ylim(0, params['L0'])
    axes[0,0].tick_params(axis='both', which='major', labelsize=9)

    axes[0,1].set_title("Temperature vs ξ (time evolution)", fontsize=12)
    axes[0,1].set_xlabel("ξ (normalized length)", fontsize=10)
    axes[0,1].set_ylabel("Temperature [K]", fontsize=10)
    axes[0,1].grid(True)
    axes[0,1].set_box_aspect(0.6)
    axes[0,1].plot(xi, T_sol[0,:], color='magenta', linewidth=1)
    axes[0,1].set_xlim(0, 1)
    axes[0,1].set_ylim(0, np.max(T_sol)*1.1)
    axes[0,1].axhline(params['T_abl'], color='red', linestyle='--', label='Ablation Onset Temperature', linewidth=0.5)
    axes[0,1].tick_params(axis='both', which='major', labelsize=9)

    axes[0,2].set_title("Surface Temperature vs Time", fontsize=12)
    axes[0,2].set_xlabel("Time [s]", fontsize=10)
    axes[0,2].set_ylabel("Temperature [K]", fontsize=10)
    axes[0,2].grid(True)
    axes[0,2].set_box_aspect(0.6)
    surface_temp = T_sol[:,-1]  # Surface temperature at ξ=1
    axes[0,2].plot(t_sol, surface_temp, color='green', linewidth=2)
    axes[0,2].set_xlim(0, np.max(t_sol))
    axes[0,2].set_ylim(params['T0'], np.max(surface_temp)*1.1)
    axes[0,2].axhline(params['T_abl'], color='red', linestyle='--', label='Ablation Onset Temperature', linewidth=0.5)
    axes[0,2].tick_params(axis='both', which='major', labelsize=9)

    # --- Bottom row subplots
    axes[1,0].set_title("Ablation Rate (-dL/dt) vs Time", fontsize=12)
    axes[1,0].set_xlabel("Time [s]", fontsize=10)
    axes[1,0].set_ylabel("-dL/dt [m/s]", fontsize=10)
    axes[1,0].grid(True)
    axes[1,0].set_box_aspect(0.6)
    ablation_rate = np.gradient(-L_sol, t_sol, edge_order=2) # Numerical derivative with finite differences.
    axes[1,0].plot(t_sol, ablation_rate, color='blue', linewidth=1)
    axes[1,0].set_xlim(0, np.max(t_sol))
    axes[1,0].set_ylim(0, np.max(ablation_rate)*1.1)
    axes[1,0].tick_params(axis='both', which='major', labelsize=9)

    axes[1,1].set_title("Temperature vs x (time evolution)", fontsize=12)
    axes[1,1].set_xlabel("x [m]", fontsize=10)
    axes[1,1].set_ylabel("Temperature [K]", fontsize=10)
    axes[1,1].grid(True)
    axes[1,1].set_box_aspect(0.6)
    x_grid = xi * L_sol[0] 
    axes[1,1].plot(x_grid, T_sol[0,:], color='magenta', linewidth=1)
    axes[1,1].set_xlim(0, params['L0'])
    axes[1,1].set_ylim(0, np.max(T_sol)*1.1)
    axes[1,1].axhline(params['T_abl'], color='red', linestyle='--', label='Ablation Onset Temperature', linewidth=0.5)
    axes[1,1].tick_params(axis='both', which='major', labelsize=9)

    # --- Bottom-right subplot: Temperature Bar with Animation
    axes[1,2].set_title("Temperature Distribution (animated)", fontsize=12)
    axes[1,2].set_xlabel("x [m]", fontsize=10)
    axes[1,2].set_yticks([])
    axes[1,2].grid(False)
    axes[1,2].set_box_aspect(0.2)
    axes[1,2].tick_params(axis='both', which='major', labelsize=9)

    # Create colormap (blue -> red, black for NaN)
    colors = ['blue', 'red']
    cmap_blue_red = LinearSegmentedColormap.from_list('blue_red', colors, N=256)
    cmap_blue_red.set_bad(color='black')  # NaN = black

    # Create initial heatmap
    x_uniform = np.linspace(0, params['L0'], 200)
    temp_uniform = np.interp(x_uniform, xi * L_sol[0], T_sol[0,:])
    temp_bar = temp_uniform.reshape(1, -1)
    im = axes[1,2].imshow(temp_bar, 
                        extent=[0, params['L0'], -0.5, 0.5],
                        aspect='auto', 
                        cmap=cmap_blue_red,
                        vmin=params['T0'],
                        vmax=np.max(T_sol),
                        interpolation='bilinear')

    # Add colorbar BELOW the heatmap
    divider = make_axes_locatable(axes[1,2])
    cax = divider.append_axes("bottom", size="20%", pad=0.3)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Temperature [K] (Black = Ablated)', fontsize=9)

    axes[1,2].set_xlim(0, params['L0'])
    axes[1,2].set_ylim(-0.5, 0.5)

    # Create single time counter in upper right
    time_text_global = fig.text(0.98, 0.95, 'Time: 00:00.000', 
                               ha='right', va='top',
                               fontsize=14, color='black', weight='bold',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

    # Animation controller with simple controls
    class ControllableStopwatchAnimation:
        def __init__(self):
            self.start_time = None
            self.paused_time = 0.0
            self.is_paused = True
            self.is_reset = True
            
        def start_stop_animation(self, event):
            """Start/Stop button callback"""
            if self.is_paused:
                # Starting or resuming
                if self.is_reset:
                    # First start
                    self.start_time = time.time()
                    self.paused_time = 0.0
                    self.is_reset = False
                else:
                    # Resuming from pause
                    self.start_time = time.time() - self.paused_time
                self.is_paused = False
                button_start.label.set_text('Pause')
            else:
                # Pausing
                self.paused_time = time.time() - self.start_time
                self.is_paused = True
                button_start.label.set_text('Start')
        
        def reset_animation(self, event):
            """Reset button callback"""
            self.start_time = None
            self.paused_time = 0.0
            self.is_paused = True
            self.is_reset = True
            button_start.label.set_text('Start')
            # Update display to initial state
            self.update_display_to_time(0.0)
            
        def update_display_to_time(self, target_time):
            """Update all plots to show data at target_time"""
            # Clamp to simulation bounds
            target_time = np.clip(target_time, 0, t_sol[-1])
            
            # Interpolate data
            current_temp, current_L = self.interpolate_data(target_time)
            
            # Update plots
            xi = np.linspace(0, 1, params['N'] + 1)
            axes[0,1].lines[0].set_data(xi, current_temp)
            
            x_phys = xi * current_L
            axes[1,1].lines[0].set_data(x_phys, current_temp)
            
            # Update heatmap
            self.update_heatmap(current_temp, current_L)
            
            # Update time display
            minutes = int(target_time // 60)
            seconds = target_time % 60
            time_string = f'Time: {minutes:02d}:{seconds:06.3f}'
            time_text_global.set_text(time_string)
            
            plt.draw()
            
        def interpolate_data(self, target_time):
            """Interpolate temperature and length"""
            target_time = np.clip(target_time, t_sol[0], t_sol[-1])
            
            L_interp_func = interp1d(t_sol, L_sol, kind='linear', fill_value='extrapolate')
            current_L = L_interp_func(target_time)
            
            current_temp = np.zeros(params['N'] + 1)
            for i in range(params['N'] + 1):
                temp_at_xi_i = T_sol[:, i]
                temp_interp_func = interp1d(t_sol, temp_at_xi_i, kind='linear', fill_value='extrapolate')
                current_temp[i] = temp_interp_func(target_time)
            
            return current_temp, current_L
        
        def update_heatmap(self, current_temp, current_L):
            """Update the heatmap display"""
            xi_coords = np.linspace(0, 1, params['N'] + 1)
            x_current = xi_coords * current_L
            
            if current_L < params['L0']:
                n_extra_points = params['N'] + 1
                x_extra = np.linspace(current_L, params['L0'], n_extra_points)[1:]
                x_row = np.concatenate([x_current, x_extra])
                temp_row = np.concatenate([current_temp, np.full(len(x_extra), np.nan)])
            else:
                x_row = x_current
                temp_row = current_temp
            
            x_uniform = np.linspace(0, params['L0'], 200)
            temp_uniform = np.interp(x_uniform, x_row, temp_row)
            temp_bar = temp_uniform.reshape(1, -1)
            im.set_array(temp_bar)
            
        def animate_temperature(self, frame_num):
            """Animation function that runs continuously"""
            if self.is_paused:
                return [im, time_text_global]
                
            # Calculate elapsed time
            current_time = time.time()
            if self.start_time is None:
                elapsed_time = 0
            else:
                elapsed_time = current_time - self.start_time
            
            # Check if we've reached the end
            if elapsed_time >= t_sol[-1]:
                # Auto-pause at end (don't reset completely)
                self.paused_time = t_sol[-1] 
                self.is_paused = True
                self.is_reset = False  # KEY FIX: Don't set to True
                button_start.label.set_text('Start')
                elapsed_time = t_sol[-1]
            
            self.update_display_to_time(elapsed_time)
            return [im, time_text_global]

    # Create the animation controller
    stopwatch_anim = ControllableStopwatchAnimation()

    # Add control buttons
    ax_start = plt.axes([0.02, 0.02, 0.08, 0.04])  # [left, bottom, width, height]
    ax_reset = plt.axes([0.12, 0.02, 0.08, 0.04])

    button_start = Button(ax_start, 'Start')
    button_reset = Button(ax_reset, 'Reset')

    button_start.on_clicked(stopwatch_anim.start_stop_animation)
    button_reset.on_clicked(stopwatch_anim.reset_animation)

    # Apply manual layout instead of tight_layout to avoid warning
    fig.subplots_adjust(left=0.067, bottom=0.12, right=0.933, top=0.88, 
                       wspace=0.45, hspace=0.5)

    # Create animation (store reference to prevent deletion warning)
    interval_ms = 50  # 20 fps
    animation = FuncAnimation(fig, stopwatch_anim.animate_temperature, frames=1000000, 
                        interval=interval_ms, blit=False, repeat=True)

    # Initialize display to t=0
    stopwatch_anim.update_display_to_time(0.0)
    
    # Show figure
    plt.show()
    
    # Keep reference to animation to prevent garbage collection
    return animation
    

# ==============================================================================
# USER INPUT SECTION
# ==============================================================================

print("="*70)
print("    HYPERSONIC LEADING EDGE HEAT TRANSFER SIMULATOR - FD-MOL")  
print("    Finite Difference Method of Lines with Ablation")
print("="*70)

# Get flight conditions
print("\nFLIGHT CONDITIONS:")
while True:
    try:
        mach_number = float(input("Enter Mach number (5-25): "))
        if 5 <= mach_number <= 25:
            params['M_inf'] = mach_number
            break
        else:
            print("Please enter Mach number between 5 and 25")
    except ValueError:
        print("Please enter a valid number")

while True:
    try:
        altitude = float(input("Enter altitude in km (20-80): ")) * 1000
        if 20000 <= altitude <= 80000:
            params['altitude'] = altitude
            break
        else:
            print("Please enter altitude between 20 and 80 km")
    except ValueError:
        print("Please enter a valid number")

while True:
    try:
        tmax = float(input("Enter simulation time in seconds (5-60): "))
        if 5 <= tmax <= 60:
            params['t_max'] = tmax
            break
        else:
            print("Please enter time between 5 and 60 seconds")
    except ValueError:
        print("Please enter a valid number")

# Get material choice
print_material_selection()

while True:
    material_choice = input("Select material (1-" + str(MATERIALS.__len__()) + "): ").strip()
    if material_choice in MATERIALS:
        update_material_properties(material_choice)
        break
    else:
        print("This materials doesn't exist. Try again.")

update_calculated_parameters()

# Display parameters
print(f"\n" + "="*70)
print("CALCULATED PARAMETERS")
print("="*70)
for key in params:
    print(f"{key}: {params[key]}" + (" manual RK4 only" if key == 'dt' and params['strategy'] != 'manual RK4' else ""))

# ==============================================================================
# SOLVE THE PROBLEM
# ==============================================================================

print(f"\nSolving heat equation with FD-MOL using strategy: {params['strategy']}...")

# Solve using FD-MOL
t_sol, T_sol, L_sol = solve_heat_equation_fd_mol(strategy=params['strategy'])

# Visualize results
if t_sol is not None:
    visualize_results(t_sol, T_sol, L_sol)
else:
    print("Numerical solution not implemented yet.")

print(f"\nSimulation complete!")