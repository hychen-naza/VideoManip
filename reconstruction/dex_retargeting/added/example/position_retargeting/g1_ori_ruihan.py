import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import pdb

# ---- inputs ----
object = "bowl" # "plate"
print(f"Object:, {object}")
if (object == "bowl"):
    grasp_euler = np.array([-0.4319, -0.6609, -0.0972])   # bowl radians
else:
    # np.array([2.3864, -0.2887, -1.5269]) #
    grasp_euler = np.array([2.1918,  3.5861, -2.4602]) #np.array([2.1918,  3.5861, -2.4602]) #np.array([2.7157,  3.9223, -2.3507]) #
# plate np.array([2.7157,  3.9223, -2.3507]) # np.array([2.7157,  3.9223, -2.3507]) #
init_euler = np.array([0.0, 0.0, 0.0])
flat_euler  = np.array([-1.5708, 0, 0])            # radians
# flat_euler  = np.array([0, 0, 0])

R_sim = R.from_euler("xyz", grasp_euler).as_matrix()
STR = np.array([
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
])
R_real_init = R.from_euler("xyz", flat_euler).as_matrix()

rotate_back = np.array([0, 0.0, 1.5708])

# I feel something wrong here. 10/13/2025 
# 1. xyz means local frame rotation, we usually use XYZ to represent the global frame rotation.
# 2. still not sure STR @ R_sim @ STR.T, I know sometimes we need to use, but never used it before
# especially you are xyz local frame rotation
# final_rot = R.from_euler("xyz", rotate_back).as_matrix() @ STR @ R_sim @ STR @ R_real_init
final_rot = STR @ R_sim @ STR @ R_real_init
final_euler = R.from_matrix(final_rot).as_euler("xyz")

print("Final Euler angles (xyz, radians):", final_euler)


# pdb.set_trace()
# R_grasp_my = R.from_euler("y", grasp_euler[2]) * R.from_euler("z", grasp_euler[1]) * R.from_euler("x", -1*grasp_euler[0])
# # R.from_euler("x", -1*grasp_euler[0]) * R.from_euler("z", grasp_euler[1]) * R.from_euler("y", grasp_euler[2]) 
# # Rz(-1.5708) @ 
# R_final = R_grasp_my * flat_rot #rotate2flat * init_rot #Rz(-1.5708) @ R_init @ R_grasp_my #Rx(init_euler[0]) #
# # final_euler = R_final.as_rotvec()   # radians
# final_euler = R_final.as_euler("xyz")
# np.set_printoptions(precision=9, suppress=True)
# print("Final rotation matrix:\n", R_final)
# print("Final Euler angles (xyz, radians):", final_euler)
