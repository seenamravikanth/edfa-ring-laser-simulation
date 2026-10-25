  
# EDFA Ring-Laser Simulation

Python-based numerical study of the steady-state behaviour of an erbium-doped fibre ring laser.

## Project Overview

This project reconstructs a numerical model based on a PSE602 course-project report. It studies optical power propagation, erbium population inversion, amplified spontaneous emission (ASE), and ring-cavity feedback.

## Model Parameters

| Parameter | Value |
|---|---:|
| Fibre length | 10 m |
| Spatial grid points | 1000 |
| Pump wavelength | 980 nm |
| Signal wavelength | 1550 nm |
| Input pump power | 0.2 W |
| Initial signal power | 1 nW |
| Effective mode area | 50 µm² |
| Confinement factor | 0.8 |
| Erbium concentration | 1 × 10²⁵ m⁻³ |
| Upper-state lifetime | 10 ms |
| Output coupling fraction | 5% |
| ASE channels | 1530, 1535, 1545, 1555 nm |

## Features

- Steady-state population-rate calculation
- Forward pump propagation
- Intracavity signal propagation
- Forward and backward ASE propagation
- Relaxed ring-round-trip update
- Relative-change convergence check
- Plots of pump, signal, ASE, and population inversion

## Requirements

- Python 3
- NumPy
- Matplotlib

Install the required libraries:

```bash
pip install numpy matplotlib
```

## How to Run

Run the simulation locally:

```bash
python edfa_ring_laser.py
```

The script generates plots for the pump power, intracavity signal, forward ASE, backward ASE, and normalized upper-state population.

## Important Notes

This is a reconstructed numerical model based on the project report. The implementation and convergence behaviour require testing before the results can be considered validated. The reported parameter set is intended to represent a below-threshold, non-lasing case.

The model uses simplified propagation and rate equations. Its physical assumptions and ring-cavity boundary conditions should be reviewed before using it for quantitative predictions.

## Reference

Thyagarajan and Ghatak, *Lasers: Fundamentals and Applications*, Chapter 12.
