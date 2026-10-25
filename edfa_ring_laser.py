"""
EDFA Ring-Laser Steady-State Simulation
Based on the PSE602 course-project report parameters.

Dependencies: numpy, matplotlib
Run: python3 edfa_ring_laser.py
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
h = 6.62607015e-34
c = 299792458.0

# Wavelengths and erbium cross-sections
pump_wavelength = 980e-9
signal_wavelength = 1550e-9
sigma_abs_signal = 2.5e-25
sigma_ems_signal = 3.0e-25
sigma_abs_pump = 2.0e-25
sigma_ems_pump = 0.0

ase_wavelengths = np.array([1530e-9, 1535e-9, 1545e-9, 1555e-9])
ase_bandwidth = 1e11
num_ase_channels = len(ase_wavelengths)

# Model parameters from the project report
fiber_length = 10.0
nz = 1000
effective_area = 50e-12
confinement_factor = 0.8
erbium_concentration = 1e25
upper_state_lifetime = 10e-3
background_loss_db_per_m = 0.005
background_loss = background_loss_db_per_m * np.log(10.0) / 10.0
input_pump_power = 0.2
output_coupling_fraction = 0.05
initial_ase_boundary_power = 1e-12

convergence_tolerance = 1e-8
max_inner_iterations = 300
max_cavity_iterations = 60
relaxation = 0.25

z = np.linspace(0.0, fiber_length, nz)
dz = z[1] - z[0]
photon_flux_denominator = h * c * effective_area


def solve_profiles_for_input(signal_input_power):
    """Calculate propagation profiles for a trial intracavity signal."""
    P_pump = np.linspace(input_pump_power, 0.9 * input_pump_power, nz)
    P_signal = np.full(nz, signal_input_power)
    P_ase_forward = np.full(
        (num_ase_channels, nz), initial_ase_boundary_power
    )
    P_ase_backward = np.full(
        (num_ase_channels, nz), initial_ase_boundary_power
    )
    inversion = np.full(nz, 0.97)

    for inner_iteration in range(1, max_inner_iterations + 1):
        # Photon flux densities
        phi_pump = (
            P_pump * pump_wavelength / photon_flux_denominator
        )
        phi_signal = (
            P_signal * signal_wavelength / photon_flux_denominator
        )
        phi_ase_forward = (
            P_ase_forward * ase_wavelengths[:, None]
            / photon_flux_denominator
        )
        phi_ase_backward = (
            P_ase_backward * ase_wavelengths[:, None]
            / photon_flux_denominator
        )

        # Transition rates per ion
        W_up = (
            sigma_abs_pump * phi_pump
            + sigma_abs_signal * phi_signal
            + np.sum(
                sigma_abs_signal
                * (phi_ase_forward + phi_ase_backward),
                axis=0,
            )
        )
        W_down = (
            sigma_ems_pump * phi_pump
            + sigma_ems_signal * phi_signal
            + np.sum(
                sigma_ems_signal
                * (phi_ase_forward + phi_ase_backward),
                axis=0,
            )
        )

        # Steady-state population inversion
        inversion_new = W_up / (
            W_up + W_down + 1.0 / upper_state_lifetime + 1e-300
        )
        N2 = inversion_new * erbium_concentration
        N1 = erbium_concentration - N2

        # Forward pump propagation
        pump_loss = (
            confinement_factor
            * (sigma_abs_pump * N1 - sigma_ems_pump * N2)
            + background_loss
        )
        pump_exponent = np.concatenate(
            ([0.0], np.cumsum(-pump_loss[:-1] * dz))
        )
        P_pump_new = input_pump_power * np.exp(
            np.clip(pump_exponent, -500.0, 500.0)
        )

        # Forward signal propagation
        signal_gain = (
            confinement_factor
            * (sigma_ems_signal * N2 - sigma_abs_signal * N1)
            - background_loss
        )
        signal_exponent = np.concatenate(
            ([0.0], np.cumsum(signal_gain[:-1] * dz))
        )
        P_signal_new = signal_input_power * np.exp(
            np.clip(signal_exponent, -500.0, 500.0)
        )

        # Spontaneous-emission source for ASE
        ase_nu = c / ase_wavelengths
        spontaneous_source = (
            2.0 * h * ase_nu[:, None] * ase_bandwidth
            * confinement_factor * sigma_ems_signal * N2[None, :]
        )

        ase_gain = (
            confinement_factor
            * (
                sigma_ems_signal * N2[None, :]
                - sigma_abs_signal * N1[None, :]
            )
            - background_loss
        )

        # Forward ASE propagation
        P_ase_forward_new = np.zeros_like(P_ase_forward)
        P_ase_forward_new[:, 0] = initial_ase_boundary_power

        for j in range(nz - 1):
            exponent = np.clip(ase_gain[:, j] * dz, -50.0, 50.0)
            P_ase_forward_new[:, j + 1] = (
                P_ase_forward_new[:, j] * np.exp(exponent)
                + spontaneous_source[:, j] * dz
            )

        # Backward ASE propagation
        P_ase_backward_new = np.zeros_like(P_ase_backward)
        P_ase_backward_new[:, -1] = initial_ase_boundary_power

        for j in range(nz - 1, 0, -1):
            exponent = np.clip(ase_gain[:, j] * dz, -50.0, 50.0)
            P_ase_backward_new[:, j - 1] = (
                P_ase_backward_new[:, j] * np.exp(exponent)
                + spontaneous_source[:, j] * dz
            )

        def profile_error(new, old):
            return float(
                np.max(
                    np.abs(new - old)
                    / np.maximum(np.abs(new), 1e-18)
                )
            )

        profile_error_max = max(
            profile_error(P_pump_new, P_pump),
            profile_error(P_signal_new, P_signal),
            profile_error(P_ase_forward_new, P_ase_forward),
            profile_error(P_ase_backward_new, P_ase_backward),
            float(np.max(np.abs(inversion_new - inversion))),
        )

        # Under-relax the profile updates
        P_pump = (
            (1.0 - relaxation) * P_pump
            + relaxation * P_pump_new
        )
        P_signal = (
            (1.0 - relaxation) * P_signal
            + relaxation * P_signal_new
        )
        P_ase_forward = (
            (1.0 - relaxation) * P_ase_forward
            + relaxation * P_ase_forward_new
        )
        P_ase_backward = (
            (1.0 - relaxation) * P_ase_backward
            + relaxation * P_ase_backward_new
        )
        inversion = (
            (1.0 - relaxation) * inversion
            + relaxation * inversion_new
        )

        # Keep the trial input fixed during the profile solve
        P_signal[0] = signal_input_power

        if profile_error_max < convergence_tolerance:
            break

    return {
        "P_pump": P_pump,
        "P_signal": P_signal,
        "P_ase_forward": P_ase_forward,
        "P_ase_backward": P_ase_backward,
        "inversion_fraction": inversion,
        "inner_iterations": inner_iteration,
        "inner_error": profile_error_max,
        "inner_converged": (
            profile_error_max < convergence_tolerance
        ),
    }


def cavity_residual(signal_input_power):
    """Evaluate the ring-cavity boundary-condition residual."""
    solution = solve_profiles_for_input(signal_input_power)
    residual = (
        (1.0 - output_coupling_fraction)
        * solution["P_signal"][-1]
        - signal_input_power
    )
    return float(residual), solution


def solve_cavity():
    """Find a non-zero cavity fixed point using a bracketed secant method."""
    low = max(1e-9, input_pump_power * 1e-3)
    f_low, _ = cavity_residual(low)

    high = input_pump_power
    f_high, _ = cavity_residual(high)

    while f_low * f_high > 0 and high < 1e4:
        high *= 2.0
        f_high, _ = cavity_residual(high)

    if f_low * f_high > 0:
        raise RuntimeError(
            "Could not bracket a non-zero cavity fixed point. "
            "Check parameters and model assumptions."
        )

    best_solution = None
    best_power = None
    best_residual = None

    for cavity_iteration in range(1, max_cavity_iterations + 1):
        if abs(f_high - f_low) > 1e-30:
            trial = high - f_high * (high - low) / (f_high - f_low)
        else:
            trial = 0.5 * (low + high)

        if not (low < trial < high):
            trial = 0.5 * (low + high)

        residual, solution = cavity_residual(trial)
        best_solution = solution
        best_power = trial
        best_residual = residual

        if abs(residual) < 1e-7:
            break

        if f_low * residual > 0:
            low, f_low = trial, residual
        else:
            high, f_high = trial, residual

    solution = best_solution
    solution["signal_input_power"] = best_power
    solution["cavity_residual_W"] = best_residual
    solution["cavity_iterations"] = cavity_iteration
    solution["cavity_converged"] = abs(best_residual) < 1e-7
    return solution


def save_results(solution):
    """Save numerical profiles, summary, and plots."""
    output_dir = Path(__file__).resolve().parent
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    P_pump = solution["P_pump"]
    P_signal = solution["P_signal"]
    P_ase_forward = solution["P_ase_forward"]
    P_ase_backward = solution["P_ase_backward"]
    inversion = solution["inversion_fraction"]

    # Export numerical profiles to CSV
    csv_path = output_dir / "edfa_profiles.csv"
    headers = [
        "z_m",
        "pump_power_W",
        "intracavity_signal_W",
        "N2_over_Ntotal",
    ]

    for wl in ase_wavelengths:
        headers.append(f"ASE_forward_{wl * 1e9:.0f}nm_W")
    for wl in ase_wavelengths:
        headers.append(f"ASE_backward_{wl * 1e9:.0f}nm_W")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for j in range(nz):
            row = [z[j], P_pump[j], P_signal[j], inversion[j]]
            row.extend(P_ase_forward[:, j])
            row.extend(P_ase_backward[:, j])
            writer.writerow(row)

    # Plot helper functions
    def plot_pump(ax):
        ax.plot(z, P_pump, linewidth=2)
        ax.set_title("Pump Power Along Fibre")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Pump power (W)")
        ax.grid(True, alpha=0.3)

    def plot_signal(ax):
        ax.plot(z, P_signal, linewidth=2)
        ax.set_title("Intracavity Signal Along Fibre")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Signal power (W)")
        ax.grid(True, alpha=0.3)

    def plot_ase_forward(ax):
        for i, wl in enumerate(ase_wavelengths):
            ax.plot(z, P_ase_forward[i], label=f"{wl * 1e9:.0f} nm")
        ax.set_title("Forward ASE Channels")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("ASE power (W)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_ase_backward(ax):
        for i, wl in enumerate(ase_wavelengths):
            ax.plot(z, P_ase_backward[i], label=f"{wl * 1e9:.0f} nm")
        ax.set_title("Backward ASE Channels")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("ASE power (W)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_inversion(ax):
        ax.plot(z, inversion, linewidth=2)
        ax.set_title("Normalized Upper-State Population")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("N2 / Ntotal")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plot_functions = [
        ("01_pump_power.png", plot_pump),
        ("02_intracavity_signal.png", plot_signal),
        ("03_forward_ASE.png", plot_ase_forward),
        ("04_backward_ASE.png", plot_ase_backward),
        ("05_population_inversion.png", plot_inversion),
    ]

    for filename, plot_function in plot_functions:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        plot_function(ax)
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=200)
        plt.close(fig)

    # Combined figure
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    axes = axes.ravel()

    plot_pump(axes[0])
    plot_signal(axes[1])
    plot_ase_forward(axes[2])
    plot_ase_backward(axes[3])
    plot_inversion(axes[4])
    axes[5].axis("off")

    axes[5].text(
        0.02,
        0.96,
        "EDFA ring-laser steady-state model\n\n"
        f"Fibre length: {fiber_length:g} m\n"
        f"Pump input: {input_pump_power:g} W\n"
        f"Output coupling: {100 * output_coupling_fraction:g}%\n"
        f"Cavity iterations: {solution['cavity_iterations']}\n"
        f"Inner iterations: {solution['inner_iterations']}\n"
        f"Cavity residual: {solution['cavity_residual_W']:.3e} W\n"
        f"Mean inversion: {np.mean(inversion):.4f}",
        va="top",
        fontsize=12,
    )

    fig.suptitle("EDFA Ring-Laser Simulation Results", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "edfa_simulation_results.png", dpi=250)
    plt.close(fig)

    # Summary file
    output_power = output_coupling_fraction * P_signal[-1]
    with (output_dir / "simulation_summary.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("EDFA ring-laser simulation summary\n")
        file.write(
            f"Cavity fixed point converged: "
            f"{solution['cavity_converged']}\n"
        )
        file.write(
            f"Inner profiles converged: "
            f"{solution['inner_converged']}\n"
        )
        file.write(
            f"Cavity residual (W): "
            f"{solution['cavity_residual_W']:.8e}\n"
        )
        file.write(
            f"Signal power at fibre input (W): {P_signal[0]:.8e}\n"
        )
        file.write(
            f"Signal power at fibre end (W): {P_signal[-1]:.8e}\n"
        )
        file.write(
            f"Estimated output-coupled signal (W): {output_power:.8e}\n"
        )
        file.write(f"Pump power at fibre input (W): {P_pump[0]:.8e}\n")
        file.write(f"Pump power at fibre end (W): {P_pump[-1]:.8e}\n")
        file.write(
            f"Mean upper-state fraction: {np.mean(inversion):.8f}\n"
        )
        file.write(
            f"Final upper-state fraction: {inversion[-1]:.8f}\n"
        )

    return output_dir, results_dir


def main():
    solution = solve_cavity()
    save_results(solution)

    P_pump = solution["P_pump"]
    P_signal = solution["P_signal"]
    inversion = solution["inversion_fraction"]

    print("\nEDFA ring-laser simulation")
    print("--------------------------")
    print(f"Cavity fixed point converged: {solution['cavity_converged']}")
    print(f"Inner profile iteration converged: {solution['inner_converged']}")
    print(f"Cavity residual: {solution['cavity_residual_W']:.6e} W")
    print(f"Outer cavity iterations: {solution['cavity_iterations']}")
    print(f"Final inner iterations: {solution['inner_iterations']}")
    print(f"Intracavity signal at fibre input: {P_signal[0]:.6e} W")
    print(f"Intracavity signal at fibre end:   {P_signal[-1]:.6e} W")
    print(
        "Output-coupled signal estimate:   "
        f"{output_coupling_fraction * P_signal[-1]:.6e} W"
    )
    print(f"Pump power at fibre input:         {P_pump[0]:.6e} W")
    print(f"Pump power at fibre end:           {P_pump[-1]:.6e} W")
    print(f"Mean upper-state fraction:         {np.mean(inversion):.6f}")
    print(f"Final upper-state fraction:        {inversion[-1]:.6f}")
    print(
        "Saved edfa_simulation_results.png, edfa_profiles.csv, "
        "simulation_summary.txt, and five plots in results/."
    )


if __name__ == "__main__":
    main()
