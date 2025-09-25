import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import webbrowser
import os

# ==============================================================================
# HYPERSONIC LEADING EDGE HEAT TRANSFER SIMULATOR
# This program simulates heat conduction in hypersonic vehicle leading edges
# using the exact analytical solution to the 1D heat equation with Neumann BC
# ==============================================================================

import numpy as np

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
        P = 22632.06 * np.exp(-0.000157 * (h - 11000))
    else:  # Upper stratosphere (approx.)
        T = 141.94 + 0.00299 * h
        P = 2488.0 * (T / 216.65) ** (-11.388)

    R = 287.052  # J/(kg·K) for air
    rho = P / (R * T)
    return T, P, rho


def get_material_properties(material_choice: str):
    """
    Approximate material properties (k, rho, cp, Tmax) with assistance from AI tools.
    Values are representative of literature ranges; users
    may update with case-specific data.
    """
    materials = {
        '1': {'name': 'Carbon-Carbon', 'k': 80.0, 'rho': 1600.0, 'cp': 1050.0, 'max_temp': 2273},
        '2': {'name': 'HfC UHTC',       'k': 25.0, 'rho': 12600.0,'cp': 280.0,  'max_temp': 4383},
        '3': {'name': 'TaC UHTC',       'k': 22.0, 'rho': 14500.0,'cp': 265.0,  'max_temp': 4314},
        '4': {'name': 'Tungsten Alloy', 'k': 150.0,'rho': 19300.0,'cp': 134.0,  'max_temp': 3695},
        '5': {'name': 'RCC',            'k': 65.0, 'rho': 1900.0, 'cp': 900.0,  'max_temp': 2033},
    }
    return materials.get(material_choice, materials['5'])  # Default: RCC


def calculate_sutton_graves_heating(M_inf, altitude, R_n=0.001):
    """
    Calculate hypersonic heat flux using simplified Sutton-Graves equation
    q"_s = C * sqrt(rho_inf/R_n) * V_inf^3
    (Simplified form without enthalpy terms - commonly used in education)
    """
    # Get atmospheric properties
    T_inf, P_inf, rho_inf = get_atmospheric_properties(altitude)
    
    # Air properties
    gamma = 1.4        # Specific heat ratio for air
    R_gas = 287.0      # Gas constant for air [J/(kg·K)]
    
    # Calculate flight velocity
    c_inf = np.sqrt(gamma * R_gas * T_inf)  # Speed of sound
    V_inf = M_inf * c_inf                   # Flight velocity
    
    # Sutton-Graves equation - simplified form (no enthalpy terms)
    C_sg = 1.74153e-4  # Sutton-Graves constant [kg^0.5/(m^0.5·s^3)]
    qs = C_sg * np.sqrt(rho_inf / R_n) * V_inf**3
    
    return qs, V_inf, T_inf, rho_inf

def analytical_solution(x_vals, t, qs, k, L, alpha, T0):
    """
    Exact analytical solution for 1D heat conduction with Neumann boundary conditions
    Based on separation of variables and superposition principle
    T(x,t) = T_particular(x,t) + T_homogeneous(x,t)
    """
    x_vals = np.asarray(x_vals)
    
    # PARTICULAR SOLUTION - handles steady-state parabolic temperature rise
    # This represents the long-term heating pattern from constant heat flux
    particular = (qs / (2 * k * L)) * x_vals**2 + (alpha * qs / (k * L)) * t
    
    # HOMOGENEOUS SOLUTION - handles initial conditions and transient effects  
    # This represents the thermal waves that decay over time
    homogeneous = T0 - (qs * L) / (6 * k)
    
    # FOURIER SERIES - represents transient thermal waves
    # Each term is a thermal wave with different frequency and decay rate
    N_terms = 50  # Use 50 terms for good accuracy
    for n in range(1, N_terms + 1):
        # Calculate coefficient for this harmonic
        An = -2 * qs * L / (k * n**2 * np.pi**2) * ((-1)**n)
        
        # Spatial variation - cosine wave pattern
        cos_nx = np.cos(n * np.pi * x_vals / L)
        
        # Time decay - higher frequencies decay faster  
        exp_decay = np.exp(-alpha * (n * np.pi / L)**2 * t)
        
        # Add this harmonic to the homogeneous solution
        homogeneous += An * cos_nx * exp_decay
    
    return particular + homogeneous

# ==============================================================================
# USER INPUT SECTION - Get flight conditions and material choice
# ==============================================================================

print("="*70)
print("    HYPERSONIC LEADING EDGE HEAT TRANSFER SIMULATOR")  
print("    Analytical Solution to 1D Heat Equation with Neumann BC")
print("="*70)

# GET FLIGHT CONDITIONS FROM USER
print("\nFLIGHT CONDITIONS:")
while True:
    try:
        mach_number = float(input("Enter Mach number (5-25): "))
        if 5 <= mach_number <= 25:
            break
        else:
            print("Please enter Mach number between 5 and 25")
    except ValueError:
        print("Please enter a valid number")

while True:
    try:
        altitude = float(input("Enter altitude in km (20-80): ")) * 1000  # Convert to meters
        if 20000 <= altitude <= 80000:
            break
        else:
            print("Please enter altitude between 20 and 80 km")
    except ValueError:
        print("Please enter a valid number")

# GET SIMULATION TIME FROM USER
print("\nSIMULATION TIME:")
while True:
    try:
        tmax = float(input("Enter simulation time in seconds (5-60): "))
        if 5 <= tmax <= 60:
            break
        else:
            print("Please enter time between 5 and 60 seconds")
    except ValueError:
        print("Please enter a valid number")

# GET MATERIAL CHOICE FROM USER
print("\nMATERIAL SELECTION:")
print("1. Carbon-Carbon Composite (lightweight, high conductivity)")
print("2. HfC - Hafnium Carbide UHTC (ultra-high temperature)")  
print("3. TaC - Tantalum Carbide UHTC (highest density)")
print("4. Tungsten Alloy (highest conductivity, heaviest)")
print("5. Reinforced Carbon-Carbon/RCC (Space Shuttle material)")

while True:
    material_choice = input("Select material (1-5): ").strip()
    if material_choice in ['1', '2', '3', '4', '5']:
        break
    else:
        print("Please enter 1, 2, 3, 4, or 5")

# ==============================================================================
# CALCULATE PHYSICS PARAMETERS
# ==============================================================================

# Calculate hypersonic heating using Sutton-Graves equation
qs, V_inf, T_inf, rho_inf = calculate_sutton_graves_heating(mach_number, altitude)

# Get material properties
material = get_material_properties(material_choice)
k = material['k']           # Thermal conductivity [W/(m·K)]
rho_mat = material['rho']   # Material density [kg/m³]  
cp = material['cp']         # Specific heat [J/(kg·K)]
material_name = material['name']
max_temp = material['max_temp']

# Calculate derived properties
alpha = k / (rho_mat * cp)  # Thermal diffusivity [m²/s]

# Physical dimensions
L = 0.10                    # Leading edge thickness [m] - 10 cm
T0 = 300.0                  # Initial temperature [K] - room temperature

# DISPLAY CALCULATED PARAMETERS
print(f"\n" + "="*70)
print("CALCULATED FLIGHT AND MATERIAL PARAMETERS")
print("="*70)
print(f"Flight Conditions (International Standard Atmosphere 1976):")
print(f"  Mach: {mach_number:.1f}")
print(f"  Altitude: {altitude/1000:.0f} km") 
print(f"  Velocity: {V_inf:.0f} m/s ({V_inf*3.6/1000:.1f} km/s)")
print(f"  Freestream temp: {T_inf:.0f} K ({T_inf-273:.0f}°C)")
print(f"  Air density: {rho_inf:.6f} kg/m³")

print(f"\nMaterial: {material_name}")
print(f"  Thermal conductivity: {k:.0f} W/(m·K)")
print(f"  Density: {rho_mat:.0f} kg/m³")
print(f"  Specific heat: {cp:.0f} J/(kg·K)")
print(f"  Thermal diffusivity: {alpha:.2e} m²/s")
print(f"  Max operating temp: {max_temp:.0f} K ({max_temp-273:.0f}°C)")

print(f"\nHeat Transfer:")
print(f"  Sutton-Graves heat flux: {qs:.2e} W/m² ({qs/1e6:.2f} MW/m²)")
print(f"  Leading edge thickness: {L*1000:.0f} mm")
print(f"  Expected heating rate (based on asymptotic analysis): {(alpha * qs / (k * L)):.0f} K/s")
print(f"  Simulation duration: {tmax:.0f} seconds")

# ==============================================================================
# SIMULATION SETUP
# ==============================================================================

# Spatial discretization - 500 points across the leading edge
x_points = 500
x = np.linspace(0, L, x_points)

# Time discretization - use user-specified time
n_frames = 150
time_vals = np.linspace(0.0, tmax, n_frames)

# Calculate temperature bounds for animation scaling
T_initial = analytical_solution(x, 0, qs, k, L, alpha, T0)
T_final = analytical_solution(x, tmax, qs, k, L, alpha, T0)
T_min = min(T_initial.min(), T0) - 100

# Ensure y-scale is big enough to show the material limit line
# Take the maximum of: final temperature + buffer, or material limit + buffer
T_max_calculated = T_final.max() + 200
T_max_material_buffer = max_temp + 300  # Add buffer above material limit
T_max = max(T_max_calculated, T_max_material_buffer)

# ==============================================================================
# CREATE VISUALIZATION
# ==============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
title = f'Mach {mach_number:.1f} Hypersonic Leading Edge - {material_name}'
fig.suptitle(title, fontsize=16, fontweight='bold')

# Force specific spacing between subplots to prevent overlap
plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing

# TOP PLOT: Temperature distribution vs position
line_total, = ax1.plot([], [], 'red', linewidth=3, label='Total Temperature T(x,t)')
line_particular, = ax1.plot([], [], 'blue', linewidth=2, linestyle='--', 
                           alpha=0.7, label='Steady Heating Component')
line_transient, = ax1.plot([], [], 'green', linewidth=2, linestyle=':', 
                          alpha=0.7, label='Transient Heating Component')

# Add material temperature limit line with label
material_limit_line = ax1.axhline(y=max_temp, color='red', linestyle='-', alpha=0.7, linewidth=2)
ax1.text(L*1000*0.5, max_temp + (T_max-T_min)*0.02, f'Material Limit: {max_temp:.0f} K', 
         ha='center', va='bottom', fontsize=10, color='red', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.8))

ax1.set_xlim(0, L*1000)  # Convert to mm for display
ax1.set_ylim(T_min, T_max)
ax1.set_xlabel('Distance from Insulated End [mm]', fontsize=12)
ax1.set_ylabel('Temperature [K]', fontsize=12)
ax1.grid(True, alpha=0.3)

# Boundary condition annotation
ax1.annotate(f'Heated End\nq" = {qs/1e6:.2f} MW/m²', xy=(0.98, 0.05), xycoords='axes fraction',
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.8))

# Information display box - positioned at top-left
info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

# Create legend positioned perfectly under info_text box using same alignment
legend = ax1.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.68), 
                   bbox_transform=ax1.transAxes, framealpha=0.9,
                   borderaxespad=0, columnspacing=1.0, handletextpad=0.5)
legend.set_zorder(1000)  # Ensure legend stays on top

# BOTTOM PLOT: 2D heatmap showing temperature distribution as colors
rod_height = 60
temp_2d = np.zeros((rod_height, len(x)))
im = ax2.imshow(temp_2d, aspect='auto', extent=[0, L*1000, -0.5, 0.5], 
                cmap='plasma', vmin=T_min, vmax=T_max, origin='lower')

ax2.set_xlim(0, L*1000)
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlabel('Distance [mm]', fontsize=12)
ax2.set_ylabel('Leading Edge\nCross-Section', fontsize=12)
ax2.set_yticks([0])
ax2.set_yticklabels([material_name])

# Temperature colorbar - positioned lower with more space to prevent text cutoff
cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.35, shrink=0.8)
cbar.set_label('Temperature [K]', fontsize=12, fontweight='bold')

# ==============================================================================
# ANIMATION FUNCTIONS
# ==============================================================================

def init():
    """Initialize animation - clear all plots"""
    line_total.set_data([], [])
    line_particular.set_data([], [])
    line_transient.set_data([], [])
    info_text.set_text('')
    return line_total, line_particular, line_transient, info_text, im

def animate(frame):
    """Update animation for each time frame"""
    t = time_vals[frame]
    
    # Temperature components
    T_particular = (qs / (2 * k * L)) * x**2 + (alpha * qs / (k * L)) * t
    T_homogeneous = T0 - (qs * L) / (6 * k)
    
    # Add Fourier series
    N_terms = 50
    for n in range(1, N_terms + 1):
        An = -2 * qs * L / (k * n**2 * np.pi**2) * ((-1)**n)
        cos_nx = np.cos(n * np.pi * x / L)
        exp_decay = np.exp(-alpha * (n * np.pi / L)**2 * t)
        T_homogeneous += An * cos_nx * exp_decay
    
    T_total = T_particular + T_homogeneous
    
    # Update line plots (convert x to mm for display)
    line_total.set_data(x*1000, T_total)
    line_particular.set_data(x*1000, T_particular)
    line_transient.set_data(x*1000, T_homogeneous)
    
    # Update heatmap
    temp_2d_frame = np.tile(T_total, (rod_height, 1))
    im.set_array(temp_2d_frame)
    
    # Physics quantities
    surface_temp = T_total[-1]  # hottest point (heated end, x=L)
    thermal_penetration = np.sqrt(alpha * t) * 1000  # mm
    
    # Safety check
    status = "SAFE" if surface_temp < max_temp else "⚠️ LIMIT EXCEEDED"
    color = "green" if surface_temp < max_temp else "red"
    
    # Info text (aligned neatly)
    info_text.set_text(
        f"Time: {t:5.1f} s\n"
        f"Surface Temp: {surface_temp:6.0f} K  ({surface_temp-273:5.0f} °C)\n"
        f"Status: {status}\n"
        f"Penetration: {thermal_penetration:6.1f} mm\n"
        f"Material Limit: {max_temp:.0f} K"
    )
    
    return line_total, line_particular, line_transient, info_text, im


# ==============================================================================
# CREATE AND DISPLAY ANIMATION
# ==============================================================================

print(f"\nCreating animation...")
ani = FuncAnimation(fig, animate, frames=n_frames, init_func=init, 
                   blit=False, interval=100, repeat=True)

# Display options for user
print(f"\n" + "="*70)
print("ANIMATION DISPLAY OPTIONS")
print("="*70)
print("1. Show live interactive animation")
print("2. Save as HTML file (for sharing/viewing later)")
print("3. Save as GIF animation")

choice = input("Enter your choice (1/2/3): ").strip()

if choice == "1":
    print("Launching live animation window...")
    plt.show()
    
elif choice == "2":
    filename = f"hypersonic_M{mach_number:.0f}_{material_name.replace(' ', '_')}.html"
    print(f"Saving HTML animation as '{filename}'...")
    html_content = ani.to_jshtml()
    with open(filename, "w") as f:
        f.write(html_content)
    print(f"HTML file saved successfully!")
    print("Opening in web browser...")
    webbrowser.open('file://' + os.path.abspath(filename))
    
elif choice == "3":
    filename = f"hypersonic_M{mach_number:.0f}_{material_name.replace(' ', '_')}.gif"
    print(f"Saving GIF animation as '{filename}'...")
    ani.save(filename, writer="pillow", fps=10)
    print(f"GIF saved successfully as '{filename}'!")
    
else:
    print("Invalid choice. Showing live animation...")
    plt.show()

plt.close()
print(f"\nSimulation complete!")
print(f"Thank you for using the Hypersonic Leading Edge Simulator!")