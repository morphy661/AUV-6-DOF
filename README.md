# AUV Project

This repository contains the related AUV modeling, simulation, and depth sensor fault detection work.

## Project Structure

- `math-model/` - AUV mathematical model, simulation utilities, sensor/fault modules, and demo scripts.
- `depth-sensor-fault-detection/` - Depth sensor fault detection model training, inference, and generated model outputs.

## Six-thruster 6-DOF model

The default actuator layout follows the KYUBIC/Tuna-Sand2 architecture:

- four horizontal thrusters (`H1`-`H4`) actively control surge, sway, and yaw;
- two vertical thrusters (`V1`-`V2`) actively control heave;
- the vehicle dynamics retain all six degrees of freedom;
- roll and pitch are passively stabilised by hydrostatic restoring moments.

Thruster commands, forces, currents, efficiencies, and fault modes therefore
use six-element arrays in the order `H1, H2, H3, H4, V1, V2`.

The complete single-thruster validation covers 13 scenarios: one nominal run
plus no-output and thrust-loss faults for each of the six thrusters. Run it with:

```powershell
cd math-model
python examples/demo_six_dof_fault_coverage.py
```

It produces a summary table, a six-axis residual table, per-scenario source
logs, and a comparison figure under `math-model/results/six_dof_fault_coverage/`.

An oracle FTC benchmark can then compare the baseline with instantaneous,
perfectly known thruster-effectiveness reallocation:

```powershell
python examples/demo_six_dof_ideal_ftc.py
```

This benchmark is a control-performance upper bound, not a deployable fault
diagnoser: the rule-based or learned detector must supply the health estimate.

## Notes

The generated dataset `depth-sensor-fault-detection/depth_fault_detection/data/simulation_dataset.pth` is not tracked because it is larger than GitHub's normal 100 MB file limit.
