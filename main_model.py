import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL CONSTANTS & MODEL PARAMETERS
# ==========================================
G_0 = 9.81            # Sea-level gravity (m/s^2)
R_E = 6.371e6         # Earth radius (m)
RHO_0 = 1.225         # Sea-level air density (kg/m^3)
H_SCALE = 8500.0      # Atmospheric scale height (m)

MASS = 200.0          # Payload mass (kg)
AREA = 0.2            # Cross-sectional area (m^2)
C_D = 0.15            # Drag coefficient
V_0 = 2000.0          # Launch velocity (m/s) ~ Mach 6
DT = 0.005            # RK4 time step size (s)

# ==========================================
# 2. EQUATIONS OF MOTION & INTEGRATOR
# ==========================================
def atmospheric_density(z):
    z_clamped = max(z, 0.0)
    return RHO_0 * np.exp(-z_clamped / H_SCALE)

def gravity(z):
    return G_0 * (R_E / (R_E + z)) ** 2

def derivatives(state):
    x, z, vx, vz = state
    rho = atmospheric_density(z)
    g = gravity(z)
    
    v = np.sqrt(vx**2 + vz**2)
    drag_factor = 0.5 * rho * C_D * AREA / MASS
    
    ax = -drag_factor * v * vx
    az = -g - drag_factor * v * vz
    return np.array([vx, vz, ax, az])

def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# ==========================================
# 3. TRAJECTORY SIMULATION WITH INTERPOLATION
# ==========================================
def simulate_trajectory(theta_deg):
    theta_rad = np.radians(theta_deg)
    vx0 = V_0 * np.cos(theta_rad)
    vz0 = V_0 * np.sin(theta_rad)
    
    state = np.array([0.0, 0.0, vx0, vz0])
    trajectory = [state.copy()]
    time_points = [0.0]
    t = 0.0
    
    prev_state = state.copy()
    prev_t = t

    while state[3] > 0 and state[1] >= 0:
        prev_state = state.copy()
        prev_t = t
        
        state = rk4_step(state, DT)
        t += DT
        
        trajectory.append(state.copy())
        time_points.append(t)
        
    # Interpolate exact apogee location where vz = 0
    if prev_state[3] != state[3]:
        alpha = prev_state[3] / (prev_state[3] - state[3])
    else:
        alpha = 0.0
        
    apogee_state = prev_state + alpha * (state - prev_state)
    apogee_time = prev_t + alpha * DT
    
    metrics = {
        "Angle (deg)": theta_deg,
        "Max Altitude (km)": round(apogee_state[1] / 1000.0, 3),
        "Downrange at Apogee (km)": round(apogee_state[0] / 1000.0, 3),
        "Time to Apogee (s)": round(apogee_time, 2),
        "Horizontal Vel at Apogee (m/s)": round(apogee_state[2], 2)
    }
    
    return np.array(trajectory), metrics

# ==========================================
# 4. RUN SIMULATION SWEEP & SAVE DATA
# ==========================================
angles = np.arange(30, 91, 1)
summary_data = []
all_trajectories = {}

for theta in angles:
    traj, summary = simulate_trajectory(theta)
    summary_data.append(summary)
    all_trajectories[theta] = traj

df_results = pd.DataFrame(summary_data)
df_results.to_csv("trajectory_results.csv", index=False)
print("Simulation complete! Data saved to 'trajectory_results.csv'.")

# ==========================================
# 5. GENERATE & DISPLAY PLOTS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

selected_angles = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
for theta in selected_angles:
    traj = all_trajectories[theta]
    axes[0].plot(traj[:, 0] / 1000.0, traj[:, 1] / 1000.0, label=f"θ = {theta}°")

axes[0].set_title("Post-Release Trajectories to Apogee", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Downrange Distance (km)")
axes[0].set_ylabel("Altitude (km)")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

ax1 = axes[1]
ax2 = ax1.twinx()

line1 = ax1.plot(df_results["Angle (deg)"], df_results["Max Altitude (km)"], 'b-o', label="Max Altitude (km)", markersize=4)
line2 = ax2.plot(df_results["Angle (deg)"], df_results["Downrange at Apogee (km)"], 'r-s', label="Downrange (km)", markersize=4)

ax1.set_xlabel("Release Launch Angle θ (deg)")
ax1.set_ylabel("Max Altitude (km)", color='b')
ax2.set_ylabel("Downrange Distance (km)", color='r')
ax1.set_title("Optimization Curves: Altitude & Downrange vs. Angle", fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center left")

plt.tight_layout()
plt.show()