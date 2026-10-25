
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
h = 6.62607015e-34
c = 299792458.0

pw = 980e-9
sw = 1550e-9

sigma_abs_signal = 2.5e-25
sigma_ems_signal = 3.0e-25
sigma_abs_pump = 2.0e-25
sigma_ems_pump = 0.0

ase_wls = np.array([1530e-9, 1535e-9, 1545e-9, 1555e-9])
delta_nu = 1e11
num_ase_channel = len(ase_wls)

conv_tol = 1e-6
max_iter = 400
output_coupling_fraction = 0.05
relaxation = 0.5

# ============================================================
# Parameters from the report
# ============================================================
fiber_length = 10.0
nz = 1000
z = np.linspace(0, fiber_length, nz)
dz = z[1] - z[0]

Aeff = 50e-12
Gamma = 0.8
Ntot = 1e25
tau = 10e-3
alpha_bg = 0.005 * np.log(10) / 10
input_pump_power = 0.2

# Initial powers
P_f = np.ones(nz) * input_pump_power
P_s = np.ones(nz) * 1e-9
P_ase_f = np.ones((num_ase_channel, nz)) * 1e-12
P_ase_b = np.ones((num_ase_channel, nz)) * 1e-12

eps = 1e-300
converged = False

# ============================================================
# Steady-state ring-cavity iteration
# ============================================================
prev_input_signal = P_s[0]

for it in range(max_iter):
    Pp = P_f.copy()
    Ps = P_s.copy()
    Paf = P_ase_f.copy()
    Pab = P_ase_b.copy()

    # Photon flux densities
    Phi_p = Pp * pw / (h * c * Aeff + eps)
    Phi_s = Ps * sw / (h * c * Aeff + eps)

    Phi_ase_f = np.array([
        Paf[i] * ase_wls[i] / (h * c * Aeff + eps)
        for i in range(num_ase_channel)
    ])
    Phi_ase_b = np.array([
        Pab[i] * ase_wls[i] / (h * c * Aeff + eps)
        for i in range(num_ase_channel)
    ])

    # Transition rates per ion
    Wp_abs = sigma_abs_pump * Phi_p
    Wp_ems = sigma_ems_pump * Phi_p
    Ws_abs = sigma_abs_signal * Phi_s
    Ws_ems = sigma_ems_signal * Phi_s

    Wase_abs = np.sum(
        sigma_abs_signal * (Phi_ase_f + Phi_ase_b), axis=0
    )
    Wase_ems = np.sum(
        sigma_ems_signal * (Phi_ase_f + Phi_ase_b), axis=0
    )

    W_up = Wp_abs + Ws_abs + Wase_abs
    W_down = Wp_ems + Ws_ems + Wase_ems

    # Steady-state populations
    N2 = W_up / (W_up + W_down + 1.0 / tau + eps) * Ntot
    N1 = Ntot - N2

    # Signal gain coefficient
    g_signal = Gamma * (
        sigma_ems_signal * N2 - sigma_abs_signal * N1
    )

    # Pump propagation: forward
    Pp_new = np.zeros_like(Pp)
    Pp_new[0] = input_pump_power

    for k in range(nz - 1):
        alpha_p_eff = Gamma * sigma_abs_pump * N1[k] + alpha_bg
        Pp_new[k + 1] = Pp_new[k] * np.exp(-alpha_p_eff * dz)

    # Signal propagation: forward
    Ps_new = np.zeros_like(Ps)
    Ps_new[0] = Ps[0]

    for k in range(nz - 1):
        gain_eff = g_signal[k] - alpha_bg
        Ps_new[k + 1] = Ps_new[k] * np.exp(
            np.clip(gain_eff * dz, -100, 100)
        )

    # ASE spontaneous-emission source
    Pspon = np.zeros((num_ase_channel, nz))

    for i, wl in enumerate(ase_wls):
        nu = c / wl
        Pspon[i] = (
            2.0 * h * nu * delta_nu * Gamma
            * sigma_ems_signal * N2 * Aeff
            / num_ase_channel
        )

    # ASE propagation: forward
    Paf_new = np.zeros_like(Paf)
    Paf_new[:, 0] = Paf[:, 0]

    for i in range(num_ase_channel):
        for k in range(nz - 1):
            gain_eff = (
                Gamma * (
                    sigma_ems_signal * N2[k]
                    - sigma_abs_signal * N1[k]
                ) - alpha_bg
            )
            exponent = np.clip(gain_eff * dz, -100, 100)
            Paf_new[i, k + 1] = (
                Paf_new[i, k] * np.exp(exponent)
                + Pspon[i, k] * dz
            )

    # ASE propagation: backward
    Pab_new = np.zeros_like(Pab)
    Pab_new[:, -1] = Pab[:, -1]

    for i in range(num_ase_channel):
        for k in range(nz - 1, 0, -1):
            gain_eff = (
                Gamma * (
                    sigma_ems_signal * N2[k]
                    - sigma_abs_signal * N1[k]
                ) - alpha_bg
            )
            exponent = np.clip(gain_eff * dz, -100, 100)
            Pab_new[i, k - 1] = (
                Pab_new[i, k] * np.exp(exponent)
                + Pspon[i, k] * dz
            )

    # Ring round-trip boundary condition with relaxed update
    P_end = Ps_new[-1]
    desired_input = (1.0 - output_coupling_fraction) * P_end
    new_input_signal = (
        (1.0 - relaxation) * Ps[0]
        + relaxation * desired_input
    )

    # Correct convergence check: compare against previous iteration
    rel_change = abs(new_input_signal - prev_input_signal) / (
        abs(prev_input_signal) + 1e-30
    )

    P_f = Pp_new
    P_s = Ps_new
    P_s[0] = new_input_signal
    P_ase_f = Paf_new
    P_ase_b = Pab_new

    if rel_change < conv_tol:
        converged = True
        print(
            f"Converged after {it + 1} iterations "
            f"(relative change = {rel_change:.3e})."
        )
        break

    prev_input_signal = new_input_signal

if not converged:
    print(
        "Warning: maximum iterations reached. "
        "The solution may not have converged."
    )

# ============================================================
# Results
# ============================================================
print("\nEDFA ring-laser simulation")
print("--------------------------")
print(f"Pump input power : {input_pump_power:.6g} W")
print(f"Pump output power: {P_f[-1]:.6g} W")
print(f"Intracavity signal at fibre end: {P_s[-1]:.6g} W")
print(f"Signal at fibre input: {P_s[0]:.6g} W")
print(f"Output coupling fraction: {output_coupling_fraction:.3f}")
print(f"Final upper-state fraction: {N2[-1] / Ntot:.6f}")

for i, wl in enumerate(ase_wls):
    print(
        f"ASE {wl * 1e9:.0f} nm: "
        f"forward={P_ase_f[i, -1]:.6g} W, "
        f"backward={P_ase_b[i, 0]:.6g} W"
    )

# ============================================================
# Five plots
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
axes = axes.ravel()

axes[0].plot(z, P_f, linewidth=2)
axes[0].set_title("Pump Power Along Fibre")
axes[0].set_xlabel("Distance (m)")
axes[0].set_ylabel("Pump power (W)")
axes[0].grid(True)

axes[1].plot(z, P_s, linewidth=2)
axes[1].set_title("Intracavity Signal Along Fibre")
axes[1].set_xlabel("Distance (m)")
axes[1].set_ylabel("Signal power (W)")
axes[1].grid(True)

for i, wl in enumerate(ase_wls):
    axes[2].plot(z, P_ase_f[i], label=f"{wl * 1e9:.0f} nm")
axes[2].set_title("Forward ASE Channels")
axes[2].set_xlabel("Distance (m)")
axes[2].set_ylabel("ASE power (W)")
axes[2].legend()
axes[2].grid(True)

for i, wl in enumerate(ase_wls):
    axes[3].plot(z, P_ase_b[i], label=f"{wl * 1e9:.0f} nm")
axes[3].set_title("Backward ASE Channels")
axes[3].set_xlabel("Distance (m)")
axes[3].set_ylabel("ASE power (W)")
axes[3].legend()
axes[3].grid(True)

axes[4].plot(z, N2 / Ntot, linewidth=2)
axes[4].set_title("Normalized Upper-State Population")
axes[4].set_xlabel("Distance (m)")
axes[4].set_ylabel("N₂ / Ntotal")
axes[4].grid(True)

axes[5].axis("off")
axes[5].text(
    0.05, 0.8,
    "Parameters from the course-project report\n"
    "Pump = 0.2 W\n"
    "Fibre length = 10 m\n"
    "Output coupling = 5%\n\n"
    f"Converged: {converged}",
    fontsize=12,
    va="top"
)

plt.tight_layout()
plt.savefig("edfa_simulation_results.png", dpi=300)
plt.show()
