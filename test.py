import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. PHYSICAL CONSTANTS & DYNAMICS MODEL
# ==========================================
G0, RE, RHO0, H_SCALE = 9.81, 6.371e6, 1.225, 8500.0
MASS, AREA, CD = 200.0, 0.2, 0.15
V0 = 2000.0
DT = 0.025  # Simulation resolution

def derivatives(st):
    x, z, vx, vz = st
    rho = RHO0 * np.exp(-max(z, 0) / H_SCALE)
    g = G0 * (RE / (RE + max(z, 0))) ** 2
    v = np.sqrt(vx**2 + vz**2)
    drag = 0.5 * rho * CD * AREA / MASS
    return np.array([vx, vz, -drag * v * vx, -g - drag * v * vz])

def rk4_step(st, dt):
    k1 = derivatives(st)
    k2 = derivatives(st + 0.5 * dt * k1)
    k3 = derivatives(st + 0.5 * dt * k2)
    k4 = derivatives(st + dt * k3)
    return st + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# ==========================================
# 2. PRECOMPUTE TRAJECTORIES FOR ALL ANGLES
# ==========================================
angles = [30, 45, 60, 75, 89, 90]
trajectories = {}
max_len = 0

for theta in angles:
    rad = np.radians(theta)
    state = np.array([0.0, 0.0, V0 * np.cos(rad), V0 * np.sin(rad)])
    
    x_list, z_list = [0.0], [0.0]
    while state[3] > 0 and state[1] >= 0:  # Ascent phase to apogee
        state = rk4_step(state, DT)
        x_list.append(state[0] / 1000.0)  # Convert to km
        z_list.append(state[1] / 1000.0)  # Convert to km
        
    trajectories[theta] = (np.array(x_list), np.array(z_list))
    if len(x_list) > max_len:
        max_len = len(x_list)

# ==========================================
# 3. HIGH-SPEED MULTI-ANGLE ANIMATION
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

lines = {}
points = {}
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(angles)))

for theta, color in zip(angles, colors):
    line, = ax.plot([], [], color=color, lw=2, label=f"θ = {theta}°")
    point, = ax.plot([], [], marker='o', color=color, ms=6)
    lines[theta] = line
    points[theta] = point

ax.set_xlim(0, 40)
ax.set_ylim(0, 60)
ax.set_xlabel("Downrange Distance (km)", fontweight='bold')
ax.set_ylabel("Altitude (km)", fontweight='bold')
ax.set_title("Simultaneous Multi-Angle Kinetic Launch (High Speed)", fontweight='bold')
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

# Control frame skipping to adjust animation speed
STEPS_PER_FRAME = 12  # Higher number = faster animation

def animate(frame):
    current_idx = frame * STEPS_PER_FRAME
    updated_artists = []
    
    for theta in angles:
        x_data, z_data = trajectories[theta]
        
        # Clamp index once projectile reaches apogee
        idx = min(current_idx, len(x_data) - 1)
        
        lines[theta].set_data(x_data[:idx + 1], z_data[:idx + 1])
        points[theta].set_data([x_data[idx]], [z_data[idx]])
        
        updated_artists.extend([lines[theta], points[theta]])
        
    return updated_artists

total_frames = (max_len // STEPS_PER_FRAME) + 15
ani = animation.FuncAnimation(
    fig, 
    animate, 
    frames=total_frames, 
    interval=10,  # 10ms render interval for ultra-fast playback
    blit=True, 
    repeat=False
)

plt.tight_layout()
plt.show()