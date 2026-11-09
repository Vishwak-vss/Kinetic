
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. MODEL PARAMETERS
G = 9.81                 # Gravitational acceleration (m/s^2)
V0 = 2100.0              # Initial velocity (m/s)
MASS = 100.0             # Payload mass (kg)
CD = 0.08                # Drag coefficient
AREA = 0.2               # Reference area (m^2)

RHO0 = 1.06              # Density at release altitude (kg/m^3)
Y0 = 1400.0              # Initial altitude (m)
H_SCALE = 8500.0         # Atmospheric scale height (m)

DT = 0.005               # RK4 time step (s)
MAX_TIME = 200.0         # Maximum simulation time (s)


# 2. ATMOSPHERIC DENSITY AND DRAG
def atmospheric_density(y):
    """
    Exponential atmospheric density model.
    Density is referenced to the initial release altitude.
    """

    return RHO0 * np.exp(-(y - Y0) / H_SCALE)


def drag_coefficient_factor(y):
    """
    k(y) = rho(y) * CD * AREA / (2 * MASS)
    """

    return atmospheric_density(y) * CD * AREA / (2 * MASS)

# 3. DIFFERENTIAL EQUATIONS
def derivatives(state):
    """
    State vector:
    [x, y, vx, vy]

    Returns:
    [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """

    x, y, vx, vy = state

    speed = np.sqrt(vx**2 + vy**2)

    k = drag_coefficient_factor(y)

    dx_dt = vx
    dy_dt = vy

    dvx_dt = -k * speed * vx
    dvy_dt = -G - k * speed * vy

    return np.array([
        dx_dt,
        dy_dt,
        dvx_dt,
        dvy_dt
    ])

# 4. FOURTH-ORDER RUNGE-KUTTA METHOD
def rk4_step(state, dt):

    k1 = derivatives(state)

    k2 = derivatives(
        state + 0.5 * dt * k1
    )

    k3 = derivatives(
        state + 0.5 * dt * k2
    )

    k4 = derivatives(
        state + dt * k3
    )

    next_state = state + (dt / 6.0) * (
        k1 + 2*k2 + 2*k3 + k4
    )

    return next_state

# 5. TRAJECTORY SIMULATION AND APOGEE DETECTION
def simulate_trajectory(theta_deg, dt=DT):

    theta = np.radians(theta_deg)

    # Initial velocity components
    vx0 = V0 * np.cos(theta)
    vy0 = V0 * np.sin(theta)

    # Initial state vector
    state = np.array([
        0.0,
        Y0,
        vx0,
        vy0
    ])

    trajectory = [state.copy()]
    time_points = [0.0]

    t = 0.0

    # Integrate until vertical velocity crosses zero
    while state[3] > 0 and t < MAX_TIME:

        previous_state = state.copy()
        previous_time = t

        state = rk4_step(state, dt)

        t += dt

        trajectory.append(state.copy())
        time_points.append(t)

    # Check whether apogee was reached
    if state[3] > 0:
        raise RuntimeError(
            f"Apogee not reached for angle {theta_deg} degrees."
        )

    # Linear interpolation at vy = 0
    previous_vy = previous_state[3]
    current_vy = state[3]

    alpha = previous_vy / (previous_vy - current_vy)

    apogee_state = (
        previous_state +
        alpha * (state - previous_state)
    )

    apogee_time = previous_time + alpha * dt

    # Include interpolated apogee in trajectory
    trajectory[-1] = apogee_state.copy()

    trajectory = np.array(trajectory)

    # Extract results
    max_altitude = apogee_state[1] / 1000
    downrange = apogee_state[0] / 1000

    metrics = {
        "Angle (deg)": theta_deg,
        "Max Altitude (km)": max_altitude,
        "Downrange at Apogee (km)": downrange,
        "Time to Apogee (s)": apogee_time,
        "Horizontal Velocity at Apogee (m/s)": apogee_state[2]
    }

    return trajectory, np.array(time_points), metrics


# 6. ANGLE SWEEP
angles = np.arange(30, 91, 1)

results = []
all_trajectories = {}

print("Running RK4 simulations...")

for angle in angles:
    trajectory, times, metrics = simulate_trajectory(angle)
    results.append(metrics)
    all_trajectories[angle] = trajectory

df_results = pd.DataFrame(results)

# Export dataset
df_results.to_csv(
    "spin_launch_simulated_data.csv",
    index=False
)

print("\nSIMULATION RESULTS")
print(df_results.iloc[::5].round(3).to_string(index=False))


# 7. REFINED ANGLE SEARCH FOR MAXIMUM DOWNRANGE

coarse_best = df_results.loc[
    df_results["Downrange at Apogee (km)"].idxmax()
]

coarse_angle = coarse_best["Angle (deg)"]

refined_angles = np.arange(
    max(30, coarse_angle - 2),
    min(90, coarse_angle + 2) + 0.001,
    0.05
)

refined_results = []

for angle in refined_angles:

    _, _, metrics = simulate_trajectory(
        angle,
        dt=DT
    )

    refined_results.append(metrics)


df_refined = pd.DataFrame(refined_results)

best_downrange = df_refined.loc[
    df_refined["Downrange at Apogee (km)"].idxmax()
]

best_altitude = df_results.loc[
    df_results["Max Altitude (km)"].idxmax()
]

print("\nOPTIMIZATION RESULTS")

print(
    f"Maximum altitude: "
    f"{best_altitude['Max Altitude (km)']:.3f} km "
    f"at {best_altitude['Angle (deg)']} degrees"
)

print(
    f"Maximum downrange at apogee: "
    f"{best_downrange['Downrange at Apogee (km)']:.3f} km "
    f"at {best_downrange['Angle (deg)']:.2f} degrees"
)


# 8. NUMERICAL CONVERGENCE TEST

print("\nNUMERICAL CONVERGENCE TEST")

convergence_results = []

for step in [0.01, 0.005, 0.0025]:

    _, _, metrics = simulate_trajectory(
        90,
        dt=step
    )

    convergence_results.append({
        "Time step (s)": step,
        "Maximum altitude (km)": metrics["Max Altitude (km)"],
        "Apogee time (s)": metrics["Time to Apogee (s)"]
    })


df_convergence = pd.DataFrame(convergence_results)

print(df_convergence.to_string(index=False))

# Relative difference between the last two calculations
h1 = df_convergence.iloc[1]["Maximum altitude (km)"]
h2 = df_convergence.iloc[2]["Maximum altitude (km)"]

relative_difference = abs(h2 - h1) / abs(h2) * 100

print(
    f"\nRelative difference between 0.005 s "
    f"and 0.0025 s: {relative_difference:.6f}%"
)


# 9. TRAJECTORY VISUALIZATION

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

selected_angles = [30, 45, 60, 75, 90]

for angle in selected_angles:

    trajectory = all_trajectories[angle]

    x_km = trajectory[:, 0] / 1000
    y_km = trajectory[:, 1] / 1000

    axes[0].plot(
        x_km,
        y_km,
        label=f"θ = {angle}°"
    )


axes[0].set_title(
    "Post-Release Trajectories to Apogee"
)

axes[0].set_xlabel(
    "Horizontal Displacement (km)"
)

axes[0].set_ylabel(
    "Altitude (km)"
)

axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()


# 10. OPTIMIZATION GRAPHS
ax1 = axes[1]
ax2 = ax1.twinx()

line1 = ax1.plot(
    df_results["Angle (deg)"],
    df_results["Max Altitude (km)"],
    "b-o",
    markersize=3,
    label="Maximum altitude"
)

line2 = ax2.plot(
    df_results["Angle (deg)"],
    df_results["Downrange at Apogee (km)"],
    "r-s",
    markersize=3,
    label="Horizontal displacement at apogee"
)

# Mark refined downrange maximum
ax2.scatter(
    best_downrange["Angle (deg)"],
    best_downrange["Downrange at Apogee (km)"],
    marker="*",
    s=150,
    label="Refined maximum downrange"
)

ax1.set_xlabel("Release angle θ (degrees)")
ax1.set_ylabel("Maximum altitude (km)", color="blue")
ax2.set_ylabel("Horizontal displacement (km)", color="red")

ax1.set_title("Optimization Curves vs Release Angle")
ax1.grid(True, linestyle="--", alpha=0.6)

lines = line1 + line2 + [ax2.collections[0]]
labels = [line.get_label() for line in lines]

ax1.legend(lines, labels, loc="center left")

plt.tight_layout()

plt.savefig(
    "kinetic_spin_launch_analysis.png",
    dpi=300
)

plt.show()
