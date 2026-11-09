import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL CONSTANTS & MODEL PARAMETERS
# ==========================================
G_0 = 9.81            # Sea-level gravitational acceleration (m/s^2)
R_E = 6.371e6         # Radius of Earth (m)
RHO_0 = 1.225         # Sea-level air density (kg/m^3)
H_SCALE = 8500.0      # Atmospheric scale height (m)

# Payload & Launch Setup (Secondary Data Benchmarks)
MASS = 200.0          # Payload mass (kg)
AREA = 0.2            # Frontal cross-sectional area (m^2)
C_D = 0.15            # Streamlined drag coefficient (dimensionless)
V_0 = 2000.0          # Initial release velocity (m/s) ~ Mach 6
DT = 0.005            # Time step size for RK4 integration (seconds)

# ==========================================
# 2. EQUATIONS OF MOTION & RK4 INTEGRATOR
# ==========================================
def atmospheric_density(z):
    """Calculates barometric air density as a function of altitude."""
    if z < 0:
        return RHO_0
    return RHO_0 * np.exp(-z / H_SCALE)

def gravity(z):
    """Gravitational acceleration accounting for altitude variation."""
    return G_0 * (R_E / (R_E + z)) ** 2

def derivatives(state):
    """
    Computes derivatives [dx/dt, dz/dt, dvx/dt, dvz/dt] for state vector [x, z, vx, vz].
    """
    x, z, vx, vz = state
    rho = atmospheric_density(z)
    g = gravity(z)
    
    # Velocity magnitude
    v = np.sqrt(vx**2 + vz**2)
    
    # Quadratic drag deceleration coefficient
    drag_factor = 0.5 * rho * C_D * AREA / MASS
    
    # Accelerations
    ax = -drag_factor * v * vx
    az = -g - drag_factor * v * vz
    
    return np.array([vx, vz, ax, az])

def rk4_step(state, dt):
    """Executes a single 4th-Order Runge-Kutta step."""
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# ==========================================
# 3. TRAJECTORY SIMULATOR FUNCTION
# ==========================================
def simulate_trajectory(theta_deg):
    theta_rad = np.radians(theta_deg)
    vx0 = V_0 * np.cos(theta_rad)
    vz0 = V_0 * np.sin(theta_rad)
    
    state = np.array([0.0, 0.0, vx0, vz0])
    trajectory = [state.copy()]
    time_points = [0.0]
    t = 0.0
    
    # Run loop until vz becomes negative
    while state[3] > 0 and state[1] >= 0:
        prev_state = state.copy()
        prev_t = t
        
        state = rk4_step(state, DT)
        t += DT
        
        trajectory.append(state.copy())
        time_points.append(t)
        
    # --- INTERPOLATION AT APOGEE (v_z = 0) ---
    # Linear fraction alpha where vz crossed zero between prev_state and state
    if prev_state[3] != state[3]:
        alpha = prev_state[3] / (prev_state[3] - state[3])
    else:
        alpha = 0.0
        
    # Interpolate exact state vector at apogee
    apogee_state = prev_state + alpha * (state - prev_state)
    apogee_time = prev_t + alpha * DT
    
    max_alt_km = apogee_state[1] / 1000.0
    downrange_km = apogee_state[0] / 1000.0
    
    metrics = {
        "Angle (deg)": theta_deg,
        "Max Altitude (km)": round(max_alt_km, 3),
        "Downrange at Apogee (km)": round(downrange_km, 3),
        "Time to Apogee (s)": round(apogee_time, 2),
        "Horizontal Vel at Apogee (m/s)": round(apogee_state[2], 2)
    }
    
    return np.array(trajectory), metrics
# ==========================================
# 4. PARAMETRIC SWEEP & DATASET GENERATION
# ==========================================
angles = np.arange(30, 91, 1)  # 30° to 90° in 1° increments
summary_data = []
all_trajectories = {}

print("Running RK4 kinetic spin-launch simulations...")
for theta in angles:
    traj, summary = simulate_trajectory(theta)
    summary_data.append(summary)
    all_trajectories[theta] = traj

# Convert generated simulation data into a Pandas DataFrame
df_results = pd.DataFrame(summary_data)

# Save simulated dataset to CSV
df_results.to_csv("spin_launch_simulated_data.csv", index=False)
print("Simulation complete! Data saved to 'spin_launch_simulated_data.csv'.\n")

# Display sample of dataset table
print("=== SIMULATED DATASET SAMPLE ===")
print(df_results.iloc[::5].to_string(index=False))

# ==========================================
# 5. DATA VISUALIZATION (PLOTS)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Trajectory Profiles for Selected Angles
selected_angles = [30, 45, 60, 75, 90]
for theta in selected_angles:
    traj = all_trajectories[theta]
    x_km = traj[:, 0] / 1000.0
    z_km = traj[:, 1] / 1000.0
    axes[0].plot(x_km, z_km, label=f"θ = {theta}°")

axes[0].set_title("Post-Release Trajectories to Apogee", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Downrange Distance (km)", fontsize=10)
axes[0].set_ylabel("Altitude (km)", fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# Plot 2: Max Altitude & Downrange vs Launch Angle
ax1 = axes[1]
ax2 = ax1.twinx()

line1 = ax1.plot(df_results["Angle (deg)"], df_results["Max Altitude (km)"], 'b-o', label="Max Altitude (km)", markersize=4)
line2 = ax2.plot(df_results["Angle (deg)"], df_results["Downrange at Apogee (km)"], 'r-s', label="Downrange (km)", markersize=4)

ax1.set_xlabel("Release Launch Angle θ (deg)", fontsize=10)
ax1.set_ylabel("Max Altitude (km)", color='b', fontsize=10)
ax2.set_ylabel("Downrange Distance (km)", color='r', fontsize=10)
ax1.set_title("Optimization Curves: Altitude & Downrange vs. Angle", fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)

# Combined Legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center left")

plt.tight_layout()
plt.savefig("kinetic_spin_launch_analysis.png", dpi=300)
plt.show()