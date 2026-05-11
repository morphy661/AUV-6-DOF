## 基本使用方法

### 1. Single IMU ESKF（单 IMU）

```bash
## IMU Simulation and ESKF Quickstart

This directory contains simulation tools and ESKF implementations for single-, dual-, and triple-IMU navigation experiments. The README below provides quick usage examples, full command-line references for the dual-IMU script, and dedicated sections explaining how to generate IMU CSV data (both clean and noisy).

**Requirements**
- Python 3.8+
- Typical packages: `numpy`, `scipy`, `matplotlib`, `pandas` (install via `pip install -r requirements.txt` if present)

**Repository layout (relevant files)**
- `src/imu_simulation/print_imu_readings.py` — generate simulated IMU CSVs (`IMU_C.csv`, `IMU_L.csv`, `IMU_R.csv`).
- `src/imu_simulation/noise_imu_data.py` — add configurable noise to IMU CSVs and write noised versions.
- `src/imu_simulation/eskf_single_imu.py` — single-IMU ESKF runner.
- `src/imu_simulation/eskf_dual_imu.py` — dual-IMU ESKF with online extrinsic calibration.
- `src/imu_simulation/eskf_triple_imu.py` — triple-IMU ESKF runner.

**Quick Start Examples**

- Single IMU (basic):

```bash
cd ~/AUV-MathModel
python3 src/imu_simulation/eskf_single_imu.py --imu IMU_C.csv
```

- Dual IMU (basic):

```bash
python3 -m src/imu_simulation/eskf_dual_imu --imu-l IMU_L.csv --imu-r IMU_R.csv
```

- Dual IMU (use ground-truth angular acceleration if available):

```bash
python3 -m src/imu_simulation/eskf_dual_imu --imu-l IMU_L.csv --imu-r IMU_R.csv --use-true-angular-acc
```

---

**Dual-IMU: Command-line arguments (summary)**

Below are the most commonly used arguments for `eskf_dual_imu.py`. For the full and current list, run `python3 src/imu_simulation/eskf_dual_imu.py --help`.

- `--imu-l`: Left IMU CSV path (default: `IMU_L.csv`).
- `--imu-r`: Right IMU CSV path (default: `IMU_R.csv`).
- `--imu-l-noised`: Optional path to a noised left IMU CSV.
- `--imu-r-noised`: Optional path to a noised right IMU CSV.
- `--imu-c`: Optional center IMU CSV (ground truth) for comparison.
- `--output`: Trajectory figure output path (default: `dual_eskf_trajectory.png`).
- `--output-residuals`: Residuals figure output path (default: `dual_eskf_residuals.png`).
- `--output-lever-arm`: Lever-arm estimation figure output path (default: `dual_eskf_lever_arm.png`).
- `--master`: Which IMU is master: `left` or `right` (default: `left`).

- Lever-arm / extrinsic options:
  - `--lever-arm-init-x/y/z`: Initial guess for lever arm components (default: `0.0`).
  - `--init-sigma-lever-arm`: Initial std for lever-arm uncertainty (m) (default: `0.5`).
  - `--lever-arm-magnitude`: If known, supply the magnitude to constrain estimation (m).
  - `--no-magnitude-constraint`: Disable magnitude constraint even if `--lever-arm-magnitude` is set.

- Noise and filter tuning:
  - `--sigma-acc`: Accelerometer noise std (m/s^2) (default: `0.1`).
  - `--sigma-gyro`: Gyroscope noise std (rad/s) (default: `0.01`).
  - `--sigma-acc-constraint`: Accelerometer constraint noise (m/s^2) (default: `0.2`).
  - `--sigma-gyro-constraint`: Gyro constraint noise (rad/s) (default: `0.02`).
  - `--sigma-lever-arm`: Lever-arm process noise std (default: `0.0001`).

- Angular acceleration options:
  - `--no-angular-acc`: Disable angular-acceleration term (ignore α×r).
  - `--use-true-angular-acc`: Use angular acceleration from CSV (requires CSV with α columns).

- GPS (surface) options:
  - `--enable-gps`: Enable GPS position updates.
  - `--gps-cutoff-time`: Time when GPS becomes unavailable (s) (default: `0.0`).
  - `--gps-underwater-duration`: Additional time GPS remains available after submersion (s).
  - `--sigma-gps-pos`: GPS position noise (m) (default: `0.01`).

- Depth sensor (underwater) options:
  - `--enable-depth`: Enable depth updates.
  - `--depth-start-time`: Time when depth sensor becomes available (s) (default: `0.0`).
  - `--sigma-depth`: Depth sensor noise (m) (default: `0.1`).
  - `--depth-seed`: Random seed for depth noise (optional).

- Heading alignment options:
  - `--enable-heading-alignment`: Enable heading alignment (hard reset at the specified time).
  - `--heading-alignment-time`: Alignment time (s) (default: `0.0`).
  - `--initial-heading-deg`: Initial heading in degrees (0=+X, 90=+Y) (default: `90`).

---

## Generating IMU data (two workflows)

This section explains how to produce IMU CSV data used by the ESKF scripts. Two separate utilities are provided:

- `print_imu_readings.py` — generate clean/simulated IMU CSVs (center, left, right).
- `noise_imu_data.py` — take existing IMU CSVs and create noised copies.

Both utilities are in `src/imu_simulation`.

### 1) Generate simulated IMU CSVs (clean)

Use `print_imu_readings.py` to generate baseline IMU CSV files. The script constructs a parameterized trajectory and simulates an IMU pair rigidly attached to the vehicle center.

By default the script writes three files to the current working directory:
- `IMU_C.csv` — center IMU (ground truth positions + sensor readings)
- `IMU_L.csv` — left IMU
- `IMU_R.csv` — right IMU

Example (run from repository root):

```bash
python3 -m src/imu_simulation/print_imu_readings
```

Notes:
- The script's `__main__` contains a `TrajectoryConfig` with default trajectory parameters (duration, dt, amplitudes, etc.). Edit the script if you want to change the trajectory parameters programmatically.
- The generated CSV files will include `time,x,y,z,ax,ay,az,wx,wy,wz` columns. If angular acceleration is included, three additional columns `alphax,alphay,alphaz` are appended.

### 2) Create noisy IMU CSVs

Once you have clean CSVs (e.g., created by `print_imu_readings.py`), run `noise_imu_data.py` to produce corrupted/noised copies. The tool reads files matching `IMU_*.csv` in the input directory and writes modified versions to the output directory.

Key arguments (supported by the script):

- `--input-dir`: Directory holding `IMU_*.csv` files (default: `.`).
- `--output-dir`: Directory to write noised files (default: `./noised`).
- `--gaussian-std-acc`: Gaussian noise std for accelerations (default: `0.01`).
- `--gaussian-std-gyro`: Gaussian noise std for gyros (default: `0.005`).
- `--walk-std-acc`: Random-walk increment std for accelerations (default: `0.001`).
- `--walk-std-gyro`: Random-walk increment std for gyros (default: `0.0005`).
- `--seed`: RNG seed for reproducibility (optional).

Example:

```bash
python3 -m src/imu_simulation/noise_imu_data \
  --input-dir . \
  --output-dir ./noised \
  --gaussian-std-acc 0.01 \
  --gaussian-std-gyro 0.005 \
  --walk-std-acc 0.001 \
  --walk-std-gyro 0.0005 \
  --seed 42
```

After successful execution the tool prints the generated filenames, e.g. `noised/IMU_L_Nois ed.csv`, `noised/IMU_R_Nois ed.csv`, etc.

---

## Tips and next steps

- To check options for any script, run it with `--help`.
- Typical workflow:
  1. Generate clean data with `print_imu_readings.py`.
  2. Create noised versions with `noise_imu_data.py` (optional).
  3. Run `eskf_dual_imu.py` (or single/triple) with the generated CSVs.

8m/s normally 4m/s(AE)