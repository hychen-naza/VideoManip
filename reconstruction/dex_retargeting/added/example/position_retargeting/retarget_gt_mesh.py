import os
import torch
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import sapien.core as sapien
from pytransform3d import rotations
from tqdm import tqdm
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from render_common import RenderBase  
from hand_robot_common import RobotHandRetargeting
from mano_layer import MANOLayer  
import viser
import shutil
import pdb 


def icp(video_object_pc, gt_object_pc):
    # Convert to Open3D point clouds
    import open3d as o3d
    video_pcd = o3d.geometry.PointCloud()
    video_pcd.points = o3d.utility.Vector3dVector(video_object_pc)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_object_pc)
    
    # Run ICP to find the transformation matrix
    threshold = 0.04 #0.02  # ICP distance threshold
    icp_result = o3d.pipelines.registration.registration_icp(
        video_pcd, gt_pcd, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Extract the 4x4 transformation matrix
    transformation_matrix = icp_result.transformation
    return transformation_matrix
    
import trimesh 


def vis_mesh(meshs):
    scene = trimesh.Scene()
    for mesh in meshs:
        scene.add_geometry(mesh)
    scene.show()


def vis_pcs(pcs):
    # Create 3D figure
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for pc in pcs:
    # Scatter plot
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], cmap='jet', marker='o')
    # Labels
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.1, 0.1)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Point Cloud Visualization")
    plt.show()


if __name__ == "__main__":

    video_metadata = [] 
    parent_folder = './inputs/pen'
    gt_object_path = os.path.join(os.environ.get("DRO_GRASP_ROOT", ""), "data/data_urdf/object/video/power_drill/gt/power_drill.stl")  # None
    input_obj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "gt_pen.obj")
    object_name = parent_folder.split('/')[2]


    if gt_object_path is not None:
        gt_object_mesh = trimesh.load_mesh(gt_object_path)
        gt_object_pc, _ = gt_object_mesh.sample(512, return_index=True) #65536

        video_object_mesh = trimesh.load_mesh(input_obj_path)
        video_object_pc, _ = video_object_mesh.sample(512, return_index=True)

        transformation_matrix = icp(video_object_pc, gt_object_pc)
        print(transformation_matrix)
        rotation_180_z = np.array([
            [-1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1]
        ])
        
        video_object_pc = (np.hstack([video_object_pc, np.ones((video_object_pc.shape[0], 1))]) @ transformation_matrix.T)[:, :3]
        video_object_pc = (np.hstack([video_object_pc, np.ones((video_object_pc.shape[0], 1))]) @ rotation_180_z.T)[:, :3]
        vis_pcs([video_object_pc, gt_object_pc])

    # Copy the file
    if gt_object_path is not None:
        video_object_mesh.apply_transform(transformation_matrix)
        video_object_mesh.apply_transform(rotation_180_z)
        # video_object_mesh.apply_transform(rotation_matrix)
        # video_object_mesh.export(output_obj_path)
        vis_mesh([video_object_mesh, gt_object_mesh])
        #pdb.set_trace()

    user_input = input("Please type something and press Enter: ")
    if (user_input == 's'):
        video_object_mesh.export(input_obj_path)

