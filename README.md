# Chrono Matrix Adjudication

Project Chrono adjudicated against closed-form references through its own system matrices, as the fourth engine in a comparison study of rigid-body dynamics engines.

At a specified state, Chrono supplies the mass matrix M and constraint Jacobian Cq (`ChSystem::WriteSystemMatrices`). The applied forces f and the velocity term Qc = −(dCq/dt)·v come from an independent derivation, and the saddle-point system

    [ M   Cqᵀ ] [ a ]   [ f  ]
    [ Cq  0   ] [ λ ] = [ Qc ]

is solved in numpy. Body accelerations and constraint forces (compared as Cqᵀλ, not λ) are then checked against closed-form references that share no symbolic algebra, mechanics library or derivation code with Chrono. The state is applied with `Setup()` + `Update()`, not `DoAssembly()`, so Chrono does not move it, and the reference is evaluated at the state read back from Chrono.

## Results

PyChrono 10.0.0 (conda build py312h98ab86c_1187), Python 3.12, Ubuntu under WSL. Sixteen probe states per case; tolerance 1e-8 relative.

| Family | Cases | Worst acceleration error | Constraint force | Status |
|---|---|---|---|---|
| Pendulum chain, N = 1–5 | 5 | 4.0e-12 | 1.3e-12 | all agree |
| Slider-crank (both dead centres; slider/crank mass to 1e6) | 10 | 1.4e-11 (8.2e-11 at dead centre) | 4.7e-9 at mass ratio 1e6 | all agree |
| Cart-pole (cart/pole mass 10 to 1e-6) | 5 | 3.2e-13 | 4.3e-13 | all agree |

The reference guard (C = 0, dC/dt = 0, d²C/dt² = 0 at every probe) holds to 5.3e-15. The analytic Qc agrees with a central difference of Chrono's own Cq to 2.5e-9. The growth of the slider-crank force error with mass ratio is consistent with the thirteen significant figures of the exported matrices.

## The `native` contrast column

The `native` column records what `DoAssembly(FULL)` followed by `GetPosDt2()` reported as the acceleration in PyChrono 10.0.0, before the fix described below. It is a contrast, not an adjudication.

- `results_chrono_*_sparseqr.txt` and `out/`: run with a direct solver (`SPARSE_QR`), so the column shows the omitted quadratic-velocity term alone. Relative error: 4–8 % on chains with N ≥ 2, 1.0–11 on the slider-crank, 0.7–2.1 on the cart-pole.
- `results_chrono_*.txt` and `out_psor_original/`: the original run under Chrono's default solver (PSOR). On the two heaviest sliders the error is 6.2–6.3 instead of 1.0; the difference is the solver's.

The solver used for this column is set by the `CHRONO_NATIVE_SOLVER` environment variable (default `SPARSE_QR`; set it to `PSOR` to reproduce the original run). The matrix method uses no Chrono solver.

## Upstream record

The findings here were diagnosed and fixed by the Chrono team; please cite the public record:

- [projectchrono/chrono#846](https://github.com/projectchrono/chrono/issues/846): acceleration-level `DoAssembly` omitted the quadratic-velocity term. Fixed in PR [#852](https://github.com/projectchrono/chrono/pull/852) (commit 23c6790769). The corrected call reports the constrained acceleration to a numerical floor of about 1e-4 times the constraint length, set by a finite-difference estimate of the velocity term.
- [projectchrono/chrono#847](https://github.com/projectchrono/chrono/issues/847): Python bindings for `ChState`/`ChStateDelta`, the system descriptor and wrench members (commit c6acd4e5a1).
- Commit f80cf02da1: `DoAssembly` now returns `ACCELERATION_INACCURATE` when its acceleration-level solve is inaccurate, with documentation on solver choice for assembly.
- PR [#856](https://github.com/projectchrono/chrono/pull/856): unit test of acceleration-level assembly on coupled mechanisms (pendulum chains, cart-pole, slider-crank through both dead centres).

## Running

Install PyChrono 10.0.0 with conda (see `chrono/environment_pychrono.txt`), plus numpy and scipy. From the repository root:

    python chrono/sweep_chrono.py --family all

Options: `--family {all,chain,slidercrank,cartpole}`, `--no-native` to skip the contrast column, `--no-fd` to skip the Qc check, `--assembly` to apply the state with `DoAssembly` instead of `Update`.

## Files

- `reference.py`, `reference_slidercrank.py`, `reference_cartpole.py`: closed-form references (numpy and scipy only), copied unchanged from the study's suite at [MechanicsDSL/mechanicsdsl](https://github.com/MechanicsDSL/mechanicsdsl) (`stress_suite/`). `systems.py` is used only by a self-test in `reference.py`.
- `chrono/adapter_chrono.py`: builds each mechanism in Chrono from point-mass bodies, `ChLinkDistance` and `ChLinkLockPrismatic`, and carries out the matrix method.
- `chrono/sweep_chrono.py`: runs the three families on the study's probe states.
- `chrono/reproduce_846.py`, `chrono/matrix_method_pendulum.py`, `chrono/check_reaction*.py`: single-pendulum reproduction of #846, the matrix method on the pendulum, and the reaction-force binding check later fixed under #847.

## Author

Noah Parsons ([ORCID 0009-0000-7224-6040](https://orcid.org/0009-0000-7224-6040)). Not affiliated with the Project Chrono team. Released under the MIT licence.
