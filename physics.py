
from config import params, materials


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