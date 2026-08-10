import os
import torch
from pathlib import Path
from scipy.stats import wasserstein_distance
import random

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
from utils import * 
import open3d as o3d
from hand_utils.hand_model import create_hand_model
import cv2
import pdb 
def my_retarget_dataset(
    input_metadata_path: str,
    hand_type: HandType,
    robot_names: List[RobotName],
    headless: bool = True,
    use_ray_tracing: bool = False,
    retarget_only: bool = False,
    robot_hand_transformation = None
):

    retargeting = RobotHandRetargeting(
        robot_names=robot_names,
        hand_type=hand_type,
        headless=headless,
        use_ray_tracing=use_ray_tracing,
        retarget_only=retarget_only
    )

    print(f"Loading dataset from {input_metadata_path}")
    data = torch.load(input_metadata_path)
    
    hand_pose = data['hand_pose']
    hand_joint = data['hand_joint']
    
    # #pdb.set_trace()
    # #hand_translation = data['hand_translation']
    # batch_size = 1
    # hand_shape = torch.rand(10)
    # mano_layer = MANOLayer(hand_type[0].name, betas=hand_shape.numpy()) #, betas=hand_shape.numpy(
    # # # vertex, joint = mano_layer(hand_pose.unsqueeze(0), hand_translation.unsqueeze(0))
    # vertex, joint = mano_layer(torch.from_numpy(hand_pose).unsqueeze(0).to(torch.float32), torch.zeros(1, 3))
    # # joint[0] = joint[0] + hand_translation[None, :]
    # # vertex[0] = vertex[0] + hand_translation[None, :]
    # pdb.set_trace()

    if robot_hand_transformation is not None:
        ones = np.ones((hand_joint.shape[0], 1))  # Shape (21, 1)
        hand_joints_homogeneous = np.hstack([hand_joint, ones])  # Shape (21, 4)
        hand_joint = np.dot(hand_joints_homogeneous, robot_hand_transformation.T)[:,:3]

    retarget_only = True #False
    if retarget_only:
        # robot_qpos = retargeting.retarget_hands(np.array(hand_pose.reshape(16, 3)), np.array(joint[0, :]))
        robot_qpos = retargeting.retarget_hands(np.array(hand_pose.reshape(16, 3)), hand_joint)
        # pdb.set_trace()
    else:
        robot_qpos = retargeting.render_and_retarget(
            vertex=np.array(vertex),
            joint=np.array(joint),
            hand_pose=np.array(hand_pose.reshape(16, 3).unsqueeze(0)),
            mano_face=mano_layer.f.cpu().numpy(),
            object_pose=None,
            fps=5,
            video_path=os.environ.get("VIDEO_OUTPUT_PATH", "video_output_0.mp4"))
    retargeting.close()
    return robot_qpos



def icp(video_object_pc, gt_object_pc):
    # Convert to Open3D point clouds
    import open3d as o3d
    video_pcd = o3d.geometry.PointCloud()
    video_pcd.points = o3d.utility.Vector3dVector(video_object_pc)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_object_pc)
    
    # Run ICP to find the transformation matrix
    threshold = 0.05 #0.02  # ICP distance threshold
    icp_result = o3d.pipelines.registration.registration_icp(
        video_pcd, gt_pcd, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Extract the 4x4 transformation matrix
    transformation_matrix = icp_result.transformation
    return transformation_matrix
    

def mesh_bound_check(mesh, idx):
    # Check size on unnormalized mesh
    bounding_box = mesh.bounds
    min_coords, max_coords = bounding_box[0], bounding_box[1]
    max_extent = max(max_coords - min_coords)
    
    volume = video_object_mesh.volume
    print(f"idx {idx}: extend {max_coords - min_coords}, volume {volume}")
    # if (max_extent > 0.5 or max_extent < 0.1 or volume < 1e-4):
    #     return False 
    # if (max_extent > 0.5 or max_extent < 0.05 or volume < 1e-4):
    #     return False 
    return True

def object_distribution_check(video_object_pc, gt_object_pc):
    # Check shape on normalized mesh
    gt_object_pc = normalize_pcs(gt_object_pc)
    video_object_pc = normalize_pcs(video_object_pc)

    hist_gt, bins_gt = compute_d2_distribution(gt_object_pc)
    hist_video, bins_video = compute_d2_distribution(video_object_pc)

    # Compute Wasserstein distance between distributions
    d2_distance = wasserstein_distance(hist_gt, hist_video)
    #print(f"d2_distance {d2_distance}")
    return True #if d2_distance < 1. else False # for spray
    return True if d2_distance < .8 else False # for phone
    return True if d2_distance < 1. else False # for wine_bottle
    # return True if d2_distance < 0.8 else False # for microphone

def normalize_scale_gt_mesh(mesh, scale):
    #pdb.set_trace()
    vertices = np.asarray(mesh.vertices)

    # Compute min and max coordinates
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    max_extent = max(max_coords - min_coords)
    # pdb.set_trace()
    # Compute scale factor to fit in [-0.1, 0.1]
    scale_factor = scale / max_extent

    # Center the object
    centroid = (min_coords + max_coords) / 2

    vertices -= centroid  # Shift vertices by centroid
    vertices *= scale_factor 
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #mesh.apply_translation(-centroid)
    # Scale the object
    # mesh.apply_scale(scale_factor)
    return mesh

if __name__ == "__main__":

    # URDF generation
    import tyro
    #args = YCBArgs(obj_dir=YCB_MODELS_DIR_PATH_STR)
    args = tyro.cli(YCBArgs)
    urdf_args = URDFArgs()

    # Robot retarget
    robot_names = [RobotName.shadow] 
    hand_type = [HandType.right] 
    hand_models = [create_hand_model('shadowhand', torch.device('cpu'))]
    objects = ['power_drill']
    #['scissors', 'lantern'] #['power_drill', 'pen', 'wine_glass', 'spray', 'tong', 'syringe'] #['power_drill', 'pen', 'wine_glass', 'spray', 'tong', 'syringe']
    obj_idx = [98]#[0, 84]
    current_dir = os.getcwd()
    gt_dir = os.path.join(os.environ.get("DRO_GRASP_ROOT", ""), "data/data_urdf/object/video")
    for object_name, idx in zip(objects, obj_idx):
        object_mesh_count = 0  
        video_metadata = [] 
        parent_folder = os.path.join(current_dir, f'vis_data/{object_name}')
        gt_object_path = os.path.join(gt_dir, f'{object_name}/{object_name}.stl') 
        folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        
        gt_object_mesh = trimesh.load_mesh(gt_object_path)
        # Set pink color for all faces of gt_object_mesh
        pink_color = [int(0.8*255), int(0.6*255), int(0.7*255), 255]
        #pdb.set_trace() 
        gt_object_mesh.visual.face_colors = np.tile(pink_color, (len(gt_object_mesh.faces), 1))
        gt_object_pc, _ = gt_object_mesh.sample(512, return_index=True)

        input_obj_path = os.path.join(parent_folder, "obj_mesh.obj")
        video_object_mesh = trimesh.load_mesh(input_obj_path)
        video_object_pc, _ = video_object_mesh.sample(512, return_index=True)

        # Load data
        input_hand_path = os.path.join(parent_folder, "hand_data.pt")
        input_hand_mesh_path = os.path.join(parent_folder, "hand_mesh.obj")
        video_hand_mesh = trimesh.load_mesh(input_hand_mesh_path)
        video_hand_pc, _ = video_hand_mesh.sample(512, return_index=True)
        # Load the hand and object combined mesh
        input_hoi_path = os.path.join(parent_folder, f"{idx}_hoi.obj")
        video_combined_mesh = trimesh.load_mesh(input_hoi_path)
    
        transformation_matrix = icp(video_object_pc, gt_object_pc)
        video_object_pc = (np.hstack([video_object_pc, np.ones((video_object_pc.shape[0], 1))]) @ transformation_matrix.T)[:, :3]
        video_hand_pc = (np.hstack([video_hand_pc, np.ones((video_hand_pc.shape[0], 1))]) @ transformation_matrix.T)[:, :3]
        video_hand_mesh.apply_transform(transformation_matrix)
        video_object_mesh.apply_transform(transformation_matrix)
        # Visualize and save the image of the hand and object mesh
        # scene = trimesh.Scene()
        # # make the hand mesh lighter blue
        # video_hand_mesh.visual.face_colors = [0.6, 0.8, 1.0, 1]  # Increased from [0.4, 0.6, 0.8, 1]
        # # scene.add_geometry(video_hand_mesh)
        # # make the object mesh light pink
        # video_object_mesh.visual.face_colors =  [0.8, 0.8, 0.8, 1] #[0.8, 0.6, 0.7, 1]
        # scene.add_geometry(video_object_mesh)
        # # scene.show()
        # vis_mesh([gt_object_mesh, video_hand_mesh])
        # pdb.set_trace()
        
        # Get retarget robot hand qpos with icp transformation matrix 
        robot_qpos = my_retarget_dataset(
            input_metadata_path=input_hand_path,
            robot_names=robot_names,
            hand_type=hand_type,
            headless=True, 
            use_ray_tracing=False,
            retarget_only=True,
            robot_hand_transformation = transformation_matrix
        )
        qpos = np.array([-2.3021e-01, -2.5093e-01, -2.1074e-01,  2.1853e+00, -3.9767e+00,
         1.1896e+00,  4.2146e-02,  8.3710e-02, -3.4906e-01,  1.5708e+00,
         1.3435e+00,  2.6295e-01,  3.4905e-01,  4.0914e-01,  1.5708e+00,
         1.5708e+00, -3.4904e-01,  1.4323e+00,  1.5708e+00,  7.6269e-06,
         7.8544e-01, -3.4906e-01,  1.0382e+00, -8.1956e-06,  5.0004e-01,
         1.4583e-01,  1.1429e+00,  2.0944e-01, -6.9812e-01,  1.5708e+00])
        qpos = torch.from_numpy(qpos).to(torch.float32)
        #qpos = torch.from_numpy(robot_qpos[robot_names[0]][0]).to(torch.float32)
        hand_mesh = hand_models[0].get_trimesh_q(qpos)['visual']
        scene1 = trimesh.Scene()
        hand_mesh.visual.face_colors = [0.4, 0.6, 0.8, 1]
        scene1.add_geometry(hand_mesh)
        # Set pink color for all faces
        # pink_color = [int(0.8*255), int(0.6*255), int(0.7*255), 255]
        pink_color = [int(0.8*255), int(0.6*255), int(0.7*255), 255]
        gt_object_mesh.visual.face_colors = np.tile(pink_color, (len(gt_object_mesh.faces), 1))
        scene1.add_geometry(gt_object_mesh)
        scene1.show()

        # Sample point clouds
        n_points = 4096  # You can adjust the number of points
        hand_models = [create_hand_model('shadowhand', torch.device('cpu'))]
        hand_model = hand_models[0]
        initial_q = hand_model.get_initial_q()
        #hand_mesh = hand_model.get_trimesh_q(initial_q)['visual']

        hand_pc, _ = hand_mesh.sample(n_points, return_index=True)
        object_pc, _ = gt_object_mesh.sample(n_points, return_index=True)

        # Create Open3D point clouds
        hand_pcd = o3d.geometry.PointCloud()
        hand_pcd.points = o3d.utility.Vector3dVector(hand_pc)
        # Set the same blue color as hand mesh [0.4, 0.6, 0.8]
        hand_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.4, 0.6, 0.8], (n_points, 1)))

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pc)
        # Set the same pink color as object mesh [0.8, 0.6, 0.7]
        object_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.8, 0.6, 0.7], (n_points, 1)))

       
        # Visualize all point clouds together
        # o3d.visualization.draw_geometries([hand_pcd, object_pcd])
