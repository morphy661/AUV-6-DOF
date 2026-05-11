#!/usr/bin/env python3
"""
扫描多IMU刚体约束观测噪声参数，寻找最优值
只调整 sigma_acc_constraint 和 sigma_gyro_constraint
"""

import numpy as np
import subprocess
import sys

def run_dual_imu(sigma_acc_constraint: float, sigma_gyro_constraint: float, noised: bool = False) -> float:
    """运行Dual IMU ESKF并返回RMSE"""
    cmd = [
        sys.executable, "-m", "src.imu_simulation.eskf_dual_imu",
        "--imu-l", "IMU_L.csv",
        "--imu-r", "IMU_R.csv",
        "--sigma-acc-constraint", str(sigma_acc_constraint),
        "--sigma-gyro-constraint", str(sigma_gyro_constraint),
    ]
    if noised:
        cmd.extend(["--imu-l-noised", "noised/IMU_L_Noised.csv"])
        cmd.extend(["--imu-r-noised", "noised/IMU_R_Noised.csv"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in result.stdout.split('\n'):
            if 'total:' in line:
                return float(line.split()[1])
    except Exception as e:
        print(f"Error: {e}")
    return float('inf')


def run_triple_imu(sigma_acc_constraint: float, sigma_gyro_constraint: float, noised: bool = False) -> float:
    """运行Triple IMU ESKF并返回RMSE"""
    cmd = [
        sys.executable, "-m", "src.imu_simulation.eskf_triple_imu",
        "--imu-c", "IMU_C.csv",
        "--imu-l", "IMU_L.csv",
        "--imu-r", "IMU_R.csv",
        "--alpha-source", "ground_truth",
        "--sigma-acc-constraint", str(sigma_acc_constraint),
        "--sigma-gyro-constraint", str(sigma_gyro_constraint),
    ]
    if noised:
        cmd.extend(["--imu-c-noised", "noised/IMU_C_Noised.csv"])
        cmd.extend(["--imu-l-noised", "noised/IMU_L_Noised.csv"])
        cmd.extend(["--imu-r-noised", "noised/IMU_R_Noised.csv"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in result.stdout.split('\n'):
            if 'total:' in line:
                return float(line.split()[1])
    except Exception as e:
        print(f"Error: {e}")
    return float('inf')


def main():
    # 扫描范围：从很大(1000)到很小(0.001)
    # 使用对数刻度
    sigma_values = [1000, 100, 10, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    
    print("=" * 80)
    print("多IMU刚体约束观测噪声参数扫描")
    print("=" * 80)
    
    # ===== 无噪声数据测试 =====
    print("\n" + "=" * 80)
    print("无噪声数据")
    print("=" * 80)
    
    print("\n--- Dual IMU ESKF ---")
    print(f"{'sigma_acc':>12} {'sigma_gyro':>12} {'RMSE (m)':>12}")
    print("-" * 40)
    
    best_dual_clean = (float('inf'), 0, 0)
    for sigma_acc in sigma_values:
        # 保持 sigma_gyro = sigma_acc / 5 的比例（典型比例）
        sigma_gyro = sigma_acc / 5
        rmse = run_dual_imu(sigma_acc, sigma_gyro, noised=False)
        print(f"{sigma_acc:>12.4f} {sigma_gyro:>12.4f} {rmse:>12.4f}")
        if rmse < best_dual_clean[0]:
            best_dual_clean = (rmse, sigma_acc, sigma_gyro)
    
    print(f"\n最优: sigma_acc={best_dual_clean[1]}, sigma_gyro={best_dual_clean[2]}, RMSE={best_dual_clean[0]:.4f} m")
    
    print("\n--- Triple IMU ESKF ---")
    print(f"{'sigma_acc':>12} {'sigma_gyro':>12} {'RMSE (m)':>12}")
    print("-" * 40)
    
    best_triple_clean = (float('inf'), 0, 0)
    for sigma_acc in sigma_values:
        sigma_gyro = sigma_acc / 5
        rmse = run_triple_imu(sigma_acc, sigma_gyro, noised=False)
        print(f"{sigma_acc:>12.4f} {sigma_gyro:>12.4f} {rmse:>12.4f}")
        if rmse < best_triple_clean[0]:
            best_triple_clean = (rmse, sigma_acc, sigma_gyro)
    
    print(f"\n最优: sigma_acc={best_triple_clean[1]}, sigma_gyro={best_triple_clean[2]}, RMSE={best_triple_clean[0]:.4f} m")
    
    # ===== 有噪声数据测试 =====
    print("\n" + "=" * 80)
    print("有噪声数据")
    print("=" * 80)
    
    print("\n--- Dual IMU ESKF ---")
    print(f"{'sigma_acc':>12} {'sigma_gyro':>12} {'RMSE (m)':>12}")
    print("-" * 40)
    
    best_dual_noised = (float('inf'), 0, 0)
    for sigma_acc in sigma_values:
        sigma_gyro = sigma_acc / 5
        rmse = run_dual_imu(sigma_acc, sigma_gyro, noised=True)
        print(f"{sigma_acc:>12.4f} {sigma_gyro:>12.4f} {rmse:>12.4f}")
        if rmse < best_dual_noised[0]:
            best_dual_noised = (rmse, sigma_acc, sigma_gyro)
    
    print(f"\n最优: sigma_acc={best_dual_noised[1]}, sigma_gyro={best_dual_noised[2]}, RMSE={best_dual_noised[0]:.4f} m")
    
    print("\n--- Triple IMU ESKF ---")
    print(f"{'sigma_acc':>12} {'sigma_gyro':>12} {'RMSE (m)':>12}")
    print("-" * 40)
    
    best_triple_noised = (float('inf'), 0, 0)
    for sigma_acc in sigma_values:
        sigma_gyro = sigma_acc / 5
        rmse = run_triple_imu(sigma_acc, sigma_gyro, noised=True)
        print(f"{sigma_acc:>12.4f} {sigma_gyro:>12.4f} {rmse:>12.4f}")
        if rmse < best_triple_noised[0]:
            best_triple_noised = (rmse, sigma_acc, sigma_gyro)
    
    print(f"\n最优: sigma_acc={best_triple_noised[1]}, sigma_gyro={best_triple_noised[2]}, RMSE={best_triple_noised[0]:.4f} m")
    
    # ===== 总结 =====
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"\nDual IMU:")
    print(f"  无噪声最优: sigma_acc={best_dual_clean[1]}, sigma_gyro={best_dual_clean[2]}, RMSE={best_dual_clean[0]:.4f} m")
    print(f"  有噪声最优: sigma_acc={best_dual_noised[1]}, sigma_gyro={best_dual_noised[2]}, RMSE={best_dual_noised[0]:.4f} m")
    print(f"\nTriple IMU:")
    print(f"  无噪声最优: sigma_acc={best_triple_clean[1]}, sigma_gyro={best_triple_clean[2]}, RMSE={best_triple_clean[0]:.4f} m")
    print(f"  有噪声最优: sigma_acc={best_triple_noised[1]}, sigma_gyro={best_triple_noised[2]}, RMSE={best_triple_noised[0]:.4f} m")


if __name__ == "__main__":
    main()

