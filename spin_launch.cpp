#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>

#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif
// =====================================================================
// 1. PHYSICAL & VEHICLE CONSTANTS
// =====================================================================
constexpr double M_VEHICLE   = 1000.0;      // Payload Mass (kg)
constexpr double RADIUS      = 0.45;        // Projectile radius (m)
constexpr double AREA        = M_PI * RADIUS * RADIUS; // Frontal Area (m^2)
constexpr double V0          = 2222.0;      // Launch Speed (~Mach 6.5, m/s)
constexpr double LATITUDE_DEG = 32.95;      // Launch Latitude (Spaceport America)

// Planetary Constants (WGS-84 & Gravitational Harmonics)
constexpr double MU_EARTH    = 3.986004418e14;  // Standard Gravitational Parameter
constexpr double R_EARTH     = 6378137.0;       // Earth Radius (m)
constexpr double J2_EARTH    = 1.08263e-3;      // J2 Oblateness Coefficient
constexpr double OMEGA_EARTH = 7.2921159e-5;    // Earth Rotation Rate (rad/s)
constexpr double G0          = 9.80665;         // Surface Gravity (m/s^2)
constexpr double R_GAS       = 8.3144626;       // Gas Constant (J/(mol*K))
constexpr double M_AIR       = 0.0289644;       // Molar Mass of Air (kg/mol)
constexpr double GAMMA       = 1.4;             // Specific Heat Ratio

// =====================================================================
// 2. ATMOSPHERIC & AERODYNAMIC FUNCTIONS
// =====================================================================
struct AtmoState {
    double rho;         // Air density (kg/m^3)
    double sound_speed; // Speed of sound (m/s)
};

AtmoState get_atmosphere_us76(double z) {
    if (z < 0.0) z = 0.0;
    if (z >= 84852.0) z = 84852.0; // Mesosphere cap

    struct Layer {
        double z_b, L_b, T_b, P_b;
    };

    const Layer layers[] = {
        {0.0,     -0.0065, 288.15, 101325.0},
        {11000.0,  0.0,    216.65, 22632.1},
        {20000.0,  0.0010, 216.65, 5474.89},
        {32000.0,  0.0028, 228.65, 868.019},
        {47000.0,  0.0,    270.65, 110.906},
        {51000.0, -0.0028, 270.65, 66.9388},
        {71000.0, -0.0020, 214.65, 3.95642}
    };

    int idx = 0;
    for (int i = 6; i >= 0; --i) {
        if (z >= layers[i].z_b) {
            idx = i;
            break;
        }
    }

    double z_b = layers[idx].z_b;
    double L_b = layers[idx].L_b;
    double T_b = layers[idx].T_b;
    double P_b = layers[idx].P_b;
    double dh  = z - z_b;

    double T, P;
    if (L_b != 0.0) {
        T = T_b + L_b * dh;
        P = P_b * std::pow(T / T_b, -G0 * M_AIR / (R_GAS * L_b));
    } else {
        T = T_b;
        P = P_b * std::exp(-G0 * M_AIR * dh / (R_GAS * T_b));
    }

    double rho = (P * M_AIR) / (R_GAS * T);
    double a   = std::sqrt(GAMMA * (R_GAS / M_AIR) * T);

    return {rho, a};
}

double get_hypersonic_Cd(double Mach, double altitude) {
    double Cd_cont;
    if (Mach < 0.8) {
        Cd_cont = 0.12;
    } else if (Mach <= 1.2) {
        Cd_cont = 0.12 + 0.28 * std::sin(M_PI * (Mach - 0.8) / 0.4);
    } else {
        Cd_cont = 0.10 + 0.30 * std::pow(1.2 / Mach, 0.9);
    }

    // Knudsen number transition for high-altitude rarefied gas dynamics
    double mean_free_path = 6.63e-8 * std::exp(altitude / 8400.0);
    double Knudsen = mean_free_path / (2.0 * RADIUS);

    if (Knudsen > 0.01) {
        double Cd_free_mol = 2.1;
        return Cd_cont + (Cd_free_mol - Cd_cont) * (Knudsen / (1.0 + Knudsen));
    }
    return Cd_cont;
}

// =====================================================================
// 3. SYSTEM DYNAMICS & NUMERICAL INTEGRATOR (RK4)
// =====================================================================
struct State {
    double x, z, vx, vz;
};

State system_derivatives(const State& s) {
    if (s.z < 0.0) return {0.0, 0.0, 0.0, 0.0};

    double v_rel = std::hypot(s.vx, s.vz);
    if (v_rel == 0.0) return {0.0, 0.0, 0.0, 0.0};

    AtmoState atmo = get_atmosphere_us76(s.z);
    double Mach = v_rel / atmo.sound_speed;
    double Cd   = get_hypersonic_Cd(Mach, s.z);

    // Non-linear Drag
    double Fd = 0.5 * atmo.rho * (v_rel * v_rel) * AREA * Cd;

    // Gravity with J2 Oblateness
    double r_mag = R_EARTH + s.z;
    double g_j2  = (MU_EARTH / (r_mag * r_mag)) * (1.0 + 1.5 * J2_EARTH * std::pow(R_EARTH / r_mag, 2));

    // Coriolis Forces
    double lat_rad = LATITUDE_DEG * M_PI / 180.0;
    double a_coriolis_z = -2.0 * OMEGA_EARTH * s.vx * std::cos(lat_rad);
    double a_coriolis_x =  2.0 * OMEGA_EARTH * s.vz * std::cos(lat_rad);

    double ax = -(Fd / M_VEHICLE) * (s.vx / v_rel) + a_coriolis_x;
    double az = -g_j2 - (Fd / M_VEHICLE) * (s.vz / v_rel) + a_coriolis_z;

    return {s.vx, s.vz, ax, az};
}

// 4th-Order Runge-Kutta Integrator Step
State rk4_step(const State& s, double dt) {
    State k1 = system_derivatives(s);
    
    State s2 = {s.x + 0.5 * dt * k1.x, s.z + 0.5 * dt * k1.z, 
                s.vx + 0.5 * dt * k1.vx, s.vz + 0.5 * dt * k1.vz};
    State k2 = system_derivatives(s2);

    State s3 = {s.x + 0.5 * dt * k2.x, s.z + 0.5 * dt * k2.z, 
                s.vx + 0.5 * dt * k2.vx, s.vz + 0.5 * dt * k2.vz};
    State k3 = system_derivatives(s3);

    State s4 = {s.x + dt * k3.x, s.z + dt * k3.z, 
                s.vx + dt * k3.vx, s.vz + dt * k3.vz};
    State k4 = system_derivatives(s4);

    return {
        s.x  + (dt / 6.0) * (k1.x  + 2.0 * k2.x  + 2.0 * k3.x  + k4.x),
        s.z  + (dt / 6.0) * (k1.z  + 2.0 * k2.z  + 2.0 * k3.z  + k4.z),
        s.vx + (dt / 6.0) * (k1.vx + 2.0 * k2.vx + 2.0 * k3.vx + k4.vx),
        s.vz + (dt / 6.0) * (k1.vz + 2.0 * k2.vz + 2.0 * k3.vz + k4.vz)
    };
}

// =====================================================================
// 4. MAIN SIMULATION & PARAMETER SWEEP
// =====================================================================
int main() {
    double dt = 0.01; // Integration time step (seconds)
    double best_angle = 0.0;
    double max_apogee = 0.0;

    std::vector<double> angles;
    std::vector<double> apogees;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Starting C++ Hypersonic Trajectory Sweep...\n";
    std::cout << "---------------------------------------------------\n";

    for (double theta = 35.0; theta <= 85.0; theta += 0.25) {
        double rad = theta * M_PI / 180.0;
        State state = {0.0, 0.0, V0 * std::cos(rad), V0 * std::sin(rad)};

        // Run until vertical velocity drops below zero (Apogee)
        while (state.vz > 0.0) {
            state = rk4_step(state, dt);
            if (state.z < 0.0) break;
        }

        double apogee_km = state.z / 1000.0;
        angles.push_back(theta);
        apogees.push_back(apogee_km);

        if (apogee_km > max_apogee) {
            max_apogee = apogee_km;
            best_angle = theta;
        }
    }

    std::cout << "\n===================================================\n";
    std::cout << "C++ SIMULATION COMPLETE\n";
    std::cout << "Optimal Release Angle (theta*) : " << best_angle << " deg\n";
    std::cout << "Maximum Orbital Apogee         : " << max_apogee << " km\n";
    std::cout << "===================================================\n";

    // Export output data to CSV for plotting
    std::ofstream csv_file("trajectory_sweep.csv");
    csv_file << "angle_deg,apogee_km\n";
    for (size_t i = 0; i < angles.size(); ++i) {
        csv_file << angles[i] << "," << apogees[i] << "\n";
    }
    csv_file.close();
    std::cout << "Data saved to 'trajectory_sweep.csv'.\n";

    return 0;
}