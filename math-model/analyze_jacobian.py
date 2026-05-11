import numpy as np

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

# When omega and alpha are near zero (stationary period)
omega = np.array([0.0001, 0.0002, 0.0001])  # nearly zero
alpha = np.array([0.0, 0.0, 0.0])  # zero

omega_cross = skew_symmetric(omega)
alpha_cross = skew_symmetric(alpha)

d_tangential_d_r = -alpha_cross
d_centripetal_d_r = omega_cross @ omega_cross

print("When omega ~ 0:")
print(f"d_tangential_d_r:\n{d_tangential_d_r}")
print(f"d_centripetal_d_r:\n{d_centripetal_d_r}")

H_lever = -(d_tangential_d_r + d_centripetal_d_r)
print(f"\nH[acc, lever] = \n{H_lever}")
print("\nPROBLEM: H is nearly zero when omega ~ 0")

# With motion
omega_motion = np.array([2.0, 1.5, 1.0])
d_centripetal_d_r_motion = skew_symmetric(omega_motion) @ skew_symmetric(omega_motion)
print(f"\nWith motion (omega = {omega_motion}):")
print(f"H[acc, lever] = \n{-d_centripetal_d_r_motion}")
