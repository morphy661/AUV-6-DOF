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

## Notes

The generated dataset `depth-sensor-fault-detection/depth_fault_detection/data/simulation_dataset.pth` is not tracked because it is larger than GitHub's normal 100 MB file limit.
