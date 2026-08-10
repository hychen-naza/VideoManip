import numpy as np
import trimesh
from urdfpy import URDF
import fpsample
import pdb 
import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
from math import pi

# Fix for numpy float deprecation warning
np.float = float

# mano_layer = ManoLayer(rot_mode="axisang",
#                         center_idx=9,
#                         mano_assets_root="<OAKINK_ROOT>/assets/mano_v1_2",
#                         use_pca=False,
#                         flat_hand_mean=True)
# hand_faces = mano_layer.th_faces  # (NF, 3)

# axisFK = AxisLayerFK(mano_assets_root="<OAKINK_ROOT>/assets/mano_v1_2")
# composed_ee = torch.zeros((1, 16, 3))

# #  transform order of right hand
# #         15-14-13-\
# #                   \
# #*   3-- 2 -- 1 -----0   < NOTE: demo on this finger
# #   6 -- 5 -- 4 ----/
# #   12 - 11 - 10 --/
# #    9-- 8 -- 7 --/

# # NOTE: the ID: 1 joints have been rotated by pi/6 around spread-axis, and pi/2 around bend-axis
# # composed_ee[:, 1] = torch.tensor([0, pi / 6, pi / 2]).unsqueeze(0)

# # # NOTE: now, the ID: 2, 3 joints have been rotated by pi/2 around bend-axis
# # composed_ee[:, 2] = torch.tensor([0, 0, pi / 2]).unsqueeze(0)
# # composed_ee[:, 3] = torch.tensor([0, 0, pi / 2]).unsqueeze(0)

# composed_aa_open = axisFK.compose(composed_ee).clone()  # (B=1, 16, 3)
# # composed_aa = torch.from_numpy(np.load('hand_pose.npy'))
# # composed_aa = composed_aa.reshape(1, -1)  # (1, 16x3)
# composed_aa_open = composed_aa_open.reshape(1,-1)
# zero_shape = torch.zeros((1, 10))
# #print(f"composed_aa {composed_aa.shape}, composed_aa_open {composed_aa_open.shape}")
# mano_output: MANOOutput = mano_layer(composed_aa_open, zero_shape)

# T_g_p = mano_output.transforms_abs  # (B=1, 16, 4, 4)
# T_g_a, R, ee = axisFK(T_g_p)
# T_g_a = T_g_a.squeeze(0)

#pdb.set_trace()

def load_robot_links_point_clouds(urdf_file, num_samples=512):
    """
    Load point clouds for each link in a robot URDF file.

    Parameters:
        urdf_file (str): Path to the URDF file.

    Returns:
        dict: A dictionary mapping link names to their point clouds.
    """
    robot = URDF.load(urdf_file)
    # robot.show() #cfg={'j_index1x':-2.0, 'j_pinky1x':2.0}
    link_point_clouds = {}
    print(len(robot.links))
    count = 0
    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry.mesh:
                    mesh_path = visual.geometry.mesh.filename
                    mesh = trimesh.load_mesh(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "panda_gripper", mesh_path))
                    #   pdb.set_trace()
                    # print(f"link {link.name} mesh_path {mesh_path}")
                    sampled_points = trimesh.sample.sample_surface(mesh, num_samples)[0]

                    # se3 = T_g_a[count].numpy()
                    # se3 = T_g_p[0, count].numpy()  # Commented out - T_g_p not defined
                    #pdb.set_trace()
                    # homogeneous_tensor = np.ones(len(sampled_points)).reshape(-1,1)
                    # link_pc_homogeneous = np.concatenate((sampled_points, homogeneous_tensor), axis=1)
                    #pdb.set_trace()
                    # link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
                    # sampled_points = link_pc_se3
                    # fig = plt.figure(figsize=(12, 8))
                    # ax = fig.add_subplot(111, projection='3d')
                    # pdb.set_trace()
                    # Plot the mesh vertices
                    # ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], c='lightgray', s=1, label='Mesh Vertices')
                    # mesh_collection = Poly3DCollection(mesh.triangles, alpha=0.5, edgecolor='k', facecolor='lightblue')
                    # ax.add_collection3d(mesh_collection)

                    # Set the aspect ratio to equal
                    # ax.set_box_aspect([1, 1, 1])
                    # Plot the sampled points
                    # ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c='red', s=10, label='Sampled Points')

                    # Set labels and legend
                    # ax.set_xlabel('X')
                    # ax.set_ylabel('Y')
                    # ax.set_zlabel('Z')
                    # ax.set_title('Mesh and Sampled Points')
                    # ax.legend()

                    # Show the plot
                    # plt.show()

                    if link.name not in link_point_clouds:
                        link_point_clouds[link.name] = sampled_points
                    else:
                        link_point_clouds[link.name] = np.vstack((
                            link_point_clouds[link.name], sampled_points
                        ))
                    print(f"link.name {link.name}")
                    count += 1

    return link_point_clouds

def extract_robot_point_cloud(urdf_file, num_samples=512):
    """
    Extract point clouds for a robot from its URDF file and apply FPS.

    Parameters:
        urdf_file (str): Path to the URDF file.
        num_samples (int): Number of points to sample using FPS.

    Returns:
        dict: A dictionary with per-link sampled point clouds and a combined FPS-sampled point cloud.
    """
    # Load point clouds for each link
    link_point_clouds = load_robot_links_point_clouds(urdf_file, num_samples=num_samples)

    # Combine all link point clouds
    combined_points = np.vstack(list(link_point_clouds.values()))

    # Apply FPS to the combined point cloud
    fps_samples_idx = fpsample.fps_sampling(combined_points, num_samples)
    fps_sampled_points = combined_points[fps_samples_idx]

    # Map FPS points to their corresponding links
    fps_points_by_link = {}
    start_idx = 0
    for link_name, points in link_point_clouds.items():
        end_idx = start_idx + len(points)
        selected_indices = [i for i in fps_samples_idx if start_idx <= i < end_idx]
        fps_points_by_link[link_name] = torch.tensor(combined_points[selected_indices], dtype=torch.float32)
        start_idx = end_idx

    # return {
    #     "link_point_clouds": link_point_clouds,
    #     "fps_sampled_points": fps_sampled_points,
    #     "fps_points_by_link": fps_points_by_link
    # }
    return {
        "original": link_point_clouds,
        "filtered": fps_points_by_link
    }

# Example usage
urdf_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "panda_gripper", "panda_gripper_glb.urdf")
result = extract_robot_point_cloud(urdf_file_path, num_samples=1012)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
points = result["filtered"]
# Plot the points
#pdb.set_trace()
points = np.vstack(list(points.values()))
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud')

# Show the plot
plt.show()
# pdb.set_trace()
torch.save(result, os.path.join(os.environ.get("DRO_GRASP_ROOT", ""), "data/PointCloud/robot/panda_gripper.pt"))
# Access per-link point clouds and FPS-sampled point cloud
# link_clouds = result["link_point_clouds"]
# fps_cloud = result["fps_sampled_points"]
# fps_cloud_by_link = result["fps_points_by_link"]
