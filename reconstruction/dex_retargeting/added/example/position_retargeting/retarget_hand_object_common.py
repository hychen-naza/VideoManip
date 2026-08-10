import os
import torch
from pathlib import Path
from scipy.stats import wasserstein_distance
import random

import numpy as np
import sapien.core as sapien
from pytransform3d import rotations
from tqdm import tqdm
# from dex_retargeting import yourdfpy as urdf
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
# from mano_layer import MANOLayer  
import viser
import shutil
import pdb 
from utils import * 
import open3d as o3d
from hand_utils.hand_model import create_hand_model

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
    threshold = 0.02 #0.02  # ICP distance threshold
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
    objects = ['my_dinner_bowl']#['power_drill', 'pen', 'microphone', 'phone', 'spray', 'wine_glass', 'tong', 'syringe', 'mug', 'longsword_large']
    obj_idx = [] #[21, 29, 66, 68, 74, 112, 116, 118]
    current_dir = os.getcwd()
    gt_dir = os.path.join(os.environ.get("DRO_GRASP_ROOT", ""), "data/data_urdf/object/video")
    for object_name in objects:
        object_mesh_count = 0  
        video_metadata = [] 
        parent_folder = os.path.join(current_dir, f'inputs/{object_name}')
        gt_object_path = os.path.join(gt_dir, f'{object_name}/obj_mesh.obj') 
        folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        #pdb.set_trace()
        for i in range(len(folders)): #len(folders)): #len(folders)
            folder = str(i)
            # Load data
            input_hand_path = os.path.join(os.path.join(parent_folder, folder), "hand_data.pt")
            input_hand_mesh_path = os.path.join(os.path.join(parent_folder, folder), "hand_mesh.obj")
            input_obj_path = os.path.join(os.path.join(parent_folder, folder), "obj_mesh.obj")
            try:
                gt_object_mesh = trimesh.load_mesh(gt_object_path)
                gt_object_pc, _ = gt_object_mesh.sample(512, return_index=True) #65536

                video_object_mesh = trimesh.load_mesh(input_obj_path)
                video_object_pc, _ = video_object_mesh.sample(512, return_index=True)

                video_hand_mesh = trimesh.load_mesh(input_hand_mesh_path)
                video_hand_pc, _ = video_hand_mesh.sample(512, return_index=True)
            except ValueError:
                continue
            print(f"i {i}, path {input_obj_path}")
            # Size and volume check, filter out too small or too large objects
            # if (not mesh_bound_check(video_object_mesh, i)):
            #     continue

            # # Shape check, filter out the less shape-liked reconstructed objects
            # if (not object_distribution_check(video_object_pc, gt_object_pc)):
            #     continue
            transformation_matrix = icp(video_object_pc, gt_object_pc)

            video_object_pc = (np.hstack([video_object_pc, np.ones((video_object_pc.shape[0], 1))]) @ transformation_matrix.T)[:, :3]
            video_hand_pc = (np.hstack([video_hand_pc, np.ones((video_hand_pc.shape[0], 1))]) @ transformation_matrix.T)[:, :3]
            
            video_hand_mesh.apply_transform(transformation_matrix)
            video_object_mesh.apply_transform(transformation_matrix)
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

            qpos = torch.from_numpy(robot_qpos[robot_names[0]][0]).to(torch.float32)
            hand_mesh = hand_models[0].get_trimesh_q(qpos)['visual']
            vis_mesh([gt_object_mesh, hand_mesh], object_name, count = i)
            
        #     output_obj_folder = f'<DRO_GRASP_ROOT>/data/data_urdf/object/video/{object_name}'
        #     output_obj_path = os.path.join(output_obj_folder, 'obj_mesh.obj')
        #     for robot_name in robot_names:
        #         meta_robot_name = ''
        #         if robot_name == RobotName.shadow:
        #             meta_robot_name = "shadowhand"
        #         qpos = torch.from_numpy(robot_qpos[robot_name][0]).to(torch.float32)
        #         video_metadata.append((qpos, f"video+{object_name}", meta_robot_name, output_obj_path))
        # torch.save(video_metadata, f"<DRO_GRASP_ROOT>/data/VideoDatasetFull/video_dataset_{object_name}.pt")

        #     user_input = input("Save this object or not, (press s for continue and any other key for continue)")
        #     if (user_input == 's'):
        #         obj_idx.append(i)
        #         # Save information
        #         output_obj_folder = f'<DRO_GRASP_ROOT>/data/data_urdf/object/video/{object_name}'
        #         output_obj_path = os.path.join(output_obj_folder, 'obj_mesh.obj')
        #         output_obj_stl_path = os.path.join(output_obj_folder, f'{object_name}.stl')
        #         output_pc_path = f'<DRO_GRASP_ROOT>/data/PointCloud/object/video/{object_name}.pt'
        #         # Create the destination directory if it doesn't exist
        #         os.makedirs(output_obj_folder, exist_ok=True)
                
        #         for robot_name in robot_names:
        #             meta_robot_name = ''
        #             if robot_name == RobotName.shadow:
        #                 meta_robot_name = "shadowhand"
        #             qpos = torch.from_numpy(robot_qpos[robot_name][0]).to(torch.float32)
        #             video_metadata.append((qpos, f"video+{object_name}", meta_robot_name, output_obj_path))
                
        #         # Generate the .obj, .stl, .urdf and point clouds for issac-gym eval

        #         # not save the reconstructed one
        #         # video_object_mesh.apply_transform(transformation_matrix)
        #         # video_object_mesh = normalize_mesh(video_object_mesh)
        #         # video_object_mesh.export(output_obj_path)
        #         # video_object_mesh.export(output_obj_stl_path)
        #         # save_mesh_pcs(video_object_mesh, output_pc_path)
        #         # process_obj_onelink(Path(output_obj_path), args.coacd_args, urdf_args)

        #         # save the gt one
        #         gt_object_mesh.export(output_obj_path)
        #         gt_object_mesh.export(output_obj_stl_path)
        #         save_mesh_pcs(gt_object_mesh, output_pc_path)
        #         process_obj_onelink(Path(output_obj_path), args.coacd_args, urdf_args)
        #         object_mesh_count += 1
        #         # pdb.set_trace()
        #         print(f"----------obj {object_name}, idx {object_mesh_count}--------")

        # print(f"obj_idx {obj_idx}")
        # torch.save(video_metadata, f"<DRO_GRASP_ROOT>/data/VideoDataset/video_dataset_{object_name}.pt")
        # torch.save(obj_idx, f"<DRO_GRASP_ROOT>/data/obj_idxs/{object_name}.pt")
        # print(f"finish saving obj {object_name}")
