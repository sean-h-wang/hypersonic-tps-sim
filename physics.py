import json
import math
import numpy as np

def get_atmospheric_properties(altitude_m: float, params):

    h = altitude_m
    
    if h < 11000:
        T = 288.19 - 0.00649 * h
        P = 101290 * (T / 288.08) ** 5.256
    elif h < 25000:
        T = 216.69
        P = 22650 * math.exp(-0.000157 * h)
    else:
        T = 141.94 + 0.00299 * h
        P = 2488.0 * (T / 216.6) ** (-11.388)

    R = params['R_gas'] # Specific gas constant for air [J/(kg·K)] # 286.9 J/(kg·K)
    rho = P / (R * T)
    return T, P, rho


def print_material_selection(materials: dict):
    """Display material selection menu - fixed for new JSON format"""
    print("\nMATERIAL SELECTION:")
    for i, (name, props) in enumerate(materials.items(), 1):
        print(f"{i}. {name} (k={props['k']} W/m·K, rho={props['rho']} kg/m³, cp={props['cp']} J/kg·K, T_abl={props['T_abl']} K)")


def update_material_properties(material_name: str, params: dict, materials:dict ):
    """Update params with selected material - fixed for new JSON format"""
    if material_name in materials:
        material = materials[material_name]
        params.update({
            'material_name': material_name,
            'k': material['k'],
            'rho': material['rho'],
            'cp': material['cp'],
            'T_abl': material['T_abl'],
            'H': material['H'],
            'epsilon': material['epsilon']
        })
    else:
        print(f"Warning: Material '{material_name}' not found.")


def calculate_sutton_graves_heating(M_inf, altitude, R_n, params: dict):
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
    T_inf, _, rho_inf = get_atmospheric_properties(altitude, params)
    
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


def calculate_stagnation_temperature(T_inf, M_inf, gamma):
    """
    Calculate stagnation temperature using Mach number relation:
    T_r = T_inf * (1 + (gamma-1)/2 * M_inf^2)
    """
    T_r = T_inf * (1 + (gamma - 1) / 2 * M_inf**2)
    return T_r


def fill_all_params(M_inf, altitude, t_max, material_id, params: dict, materials: dict):
    """
    Args:
        M_inf: Free-stream Mach number
        altitude: Altitude in KILOMETERS
        t_max: Maximum simulation time in seconds
        material_id: Material ID for TPS material (MUST BE ONE INDEXED)
        params: Dictionary to populate with all physical parameters
        materials: Dictionary of available materials loaded from JSON
    """

    # MATERIAL ID MUST BE ZERO INDEXED

    mat = materials[list(materials.keys())[material_id]]
    params['material_name'] = list(materials.keys())[material_id]
    for key in ['k','rho','cp','T_abl','H','epsilon']:
        params[key] = mat[key]

    # Update flight conditions
    altitude *= 1000
    params['altitude'] = altitude
    params['M_inf'] = M_inf
    params['t_max'] = t_max

    # Calculate derived quantities, e.g., T_inf, qs, V_inf, etc.
    # Update params in-place
    T_inf, P_inf, rho_inf = get_atmospheric_properties(params['altitude'], params)
    params['T_inf'] = T_inf
    params['P_inf'] = P_inf
    params['rho_inf'] = rho_inf
    qs, V_inf = calculate_sutton_graves_heating(M_inf, params['altitude'], params['r'], params)
    params['qs'] = qs
    params['V_inf'] = V_inf
    T_r = calculate_stagnation_temperature(T_inf, M_inf, params['gamma'])
    params['T_r'] = T_r
    params['h'] = 1 / params['N']  # Grid spacing on normalized domain [0,1]

    with open('config/params.json', 'w') as f:
        json.dump(params, f, indent=2)