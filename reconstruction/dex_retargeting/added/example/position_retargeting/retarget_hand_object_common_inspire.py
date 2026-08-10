import os
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation
import random
from typing import List
import glob
import numpy as np
from tqdm import tqdm
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from hand_robot_common import RobotHandRetargeting
# from mano_layer import MANOLayer  
import pdb 
from utils import * 
from hand_utils.hand_model import create_hand_model
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def compute_rotation_between_orientations(ori_A, ori_B):
    """
    Compute rotation matrix R such that ori_A = ori_B * inv(R)
    
    Args:
        ori_A: orientation in frame A (axis-angle format)
        ori_B: orientation in frame B (axis-angle format)
    
    Returns:
        rotation_matrix: 3x3 rotation matrix that transforms from frame B to frame A
    """
    # Convert axis-angle to rotation matrices
    rot_A = Rotation.from_rotvec(ori_A).as_matrix()  # Frame A orientation
    rot_B = Rotation.from_rotvec(ori_B).as_matrix()  # Frame B orientation
    # create a rotation matrix [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    rot_frame_B2A = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) 
    # Compute R such that ori_A = ori_B * inv(R)
    # This means: R = inv(rot_B) * rot_A
    rotation_matrix = rot_A @ np.linalg.inv(rot_B) @ np.linalg.inv(rot_frame_B2A)
    
    return rotation_matrix

def compute_rotation_in_A_frame(rotation_matrix, ori_B):
    """
    Compute rotation matrix R such that ori_A = ori_B * inv(R)
    
    Args:
        ori_A: orientation in frame A (axis-angle format)
        ori_B: orientation in frame B (axis-angle format)
    
    Returns:
        rotation_matrix: 3x3 rotation matrix that transforms from frame B to frame A
    """
    # Convert axis-angle to rotation matrices
    rot_B = Rotation.from_rotvec(ori_B).as_matrix()  # Frame B orientation
    # create a rotation matrix [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    rot_frame_B2A = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) 
    # Compute R such that ori_A = ori_B * inv(R)
    # This means: R = inv(rot_B) * rot_A
    rotation_matrix = rotation_matrix @ rot_frame_B2A @ rot_B
    # transform to axis-angle
    axis_angle = Rotation.from_matrix(rotation_matrix).as_rotvec()
    return axis_angle


def compute_orientation_difference(ori_A, ori_B):
    """
    Alternative: Compute the rotation difference between two orientations
    
    Args:
        ori_A: orientation in frame A (axis-angle format)
        ori_B: orientation in frame B (axis-angle format)
    
    Returns:
        rotation_matrix: 3x3 rotation matrix that transforms from frame B to frame A
        axis_angle_diff: axis-angle representation of the difference
    """
    # Convert axis-angle to rotation matrices
    rot_A = Rotation.from_rotvec(ori_A).as_matrix()
    rot_B = Rotation.from_rotvec(ori_B).as_matrix()
    
    # Compute the rotation that takes frame B to frame A
    # R = rot_A * rot_B.T
    rotation_matrix = rot_A @ rot_B.T
    
    # Convert to axis-angle for easier interpretation
    axis_angle_diff = Rotation.from_matrix(rotation_matrix).as_rotvec()
    
    return rotation_matrix, axis_angle_diff


def apply_forward_transform_to_qpos(qpos, forward_transform):
    """
    Apply forward transform to qpos[:6] where:
    - qpos[:3] is position (x, y, z)
    - qpos[3:6] is orientation (axis-angle format)
    
    Args:
        qpos: Robot joint positions including base pose
        forward_transform: 4x4 transformation matrix
    
    Returns:
        Transformed qpos with forward transform applied to base pose
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Extract base pose (first 6 elements)
    base_pos = qpos[:3].numpy()  # Position
    base_ori = qpos[3:6].numpy()  # Orientation (axis-angle format)
    
    # Convert axis-angle to rotation matrix
    rotation = R.from_rotvec(base_ori).as_matrix()
    
    # Create 4x4 transformation matrix from current pose
    current_transform = np.eye(4)
    current_transform[:3, :3] = rotation
    current_transform[:3, 3] = base_pos
    
    # Apply forward transformation
    # T_forward * T_current = T_result
    result_transform = forward_transform @ current_transform
    
    # Extract new position and orientation
    new_pos = result_transform[:3, 3]
    new_ori_matrix = result_transform[:3, :3]
    new_ori = R.from_matrix(new_ori_matrix).as_rotvec()  # Convert back to axis-angle
    
    # Create new qpos with transformed base pose
    new_qpos = qpos.clone()
    new_qpos[:3] = torch.from_numpy(new_pos)
    new_qpos[3:6] = torch.from_numpy(new_ori)
    
    return new_qpos

def apply_inverse_transform_to_wrist_orientation(wrist_orientation, inverse_transform):
    """
    Apply inverse transform to wrist orientation (axis-angle format)
    
    Args:
        wrist_orientation: Wrist orientation in axis-angle format (3,)
        inverse_transform: 4x4 inverse transformation matrix
    
    Returns:
        Transformed wrist orientation in axis-angle format
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Convert axis-angle to rotation matrix
    rotation = R.from_rotvec(wrist_orientation).as_matrix()
    
    # Create 4x4 transformation matrix from current orientation
    current_transform = np.eye(4)
    current_transform[:3, :3] = rotation
    
    # Apply inverse transformation to orientation only
    # T_inv * T_current = T_result
    result_transform = inverse_transform @ current_transform
    
    # Extract new orientation
    new_ori_matrix = result_transform[:3, :3]
    new_ori = R.from_matrix(new_ori_matrix).as_rotvec()  # Convert back to axis-angle
    
    return new_ori

def create_y_axis_rotation_around_origin(angle_degrees):
    """
    Create a 4x4 transformation matrix for rotation around y-axis at the origin
    
    Args:
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        4x4 transformation matrix
    """
    import numpy as np
    import math
    
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    
    # Rotation matrix around y-axis at origin
    rotation_matrix = np.array([
        [cos_angle,  0, sin_angle, 0],
        [0,          1, 0,         0],
        [-sin_angle, 0, cos_angle, 0],
        [0,          0, 0,         1]
    ])
    
    return rotation_matrix



def my_retarget_dataset(
    input_metadata: dict,
    hand_type: HandType,
    robot_names: List[RobotName],
    headless: bool = True,
    use_ray_tracing: bool = False,
    retarget_only: bool = False
):

    retargeting = RobotHandRetargeting(
        robot_names=robot_names,
        hand_type=hand_type,
        headless=headless,
        use_ray_tracing=use_ray_tracing,
        retarget_only=retarget_only
    )

    hand_pose = input_metadata['hand_pose'][:3] # only need the global orientation
    hand_joint = input_metadata['hand_joint'] 
    # pdb.set_trace()
    robot_qpos = retargeting.retarget_hands(np.array(hand_pose.reshape(-1, 3)), hand_joint)

    retargeting.close()
    return robot_qpos

if __name__ == "__main__":

    import tyro
    import argparse
    import sys
    # Parse retargeting arguments first (before tyro to avoid conflicts)
    parser = argparse.ArgumentParser(description='Hand retargeting arguments', add_help=False)
    parser.add_argument('--objects', nargs='+', default='burger_box', 
                        help='List of object names to process')
    parser.add_argument('--use-optimize', type=str, default='true',
                        choices=['true', 'false', 'True', 'False', '1', '0'],
                        help='Whether to use optimized hand parameters (default: true)')
    retarget_args, remaining_args = parser.parse_known_args()
    # Convert string to boolean
    retarget_args.use_optimize = retarget_args.use_optimize.lower() in ['true', '1']
    
    # Now parse tyro arguments with remaining args (excluding --objects and --use-optimize)
    sys.argv = [sys.argv[0]] + remaining_args
    args = tyro.cli(YCBArgs)
    urdf_args = URDFArgs()
    # for object_name in ['book_pick1', 'book_pick2', 'book_pick3', 'book_pick4', 'book_pick5']: #['pick1', 'pick2', 'pick3', 'pick4', 'pick5', 'pick6', 'pick7', 'pick8', 'pick9', 'pick10']:
    #    process_obj_onelink(Path(f"<data_root>/{object_name}/mesh/textured_simple.obj"), args.coacd_args, urdf_args)
    # exit(0)

    # Robot retarget
    # [RobotName.inspire] #
    robot_names = [RobotName.inspire] #[RobotName.leap] #[RobotName.leap, RobotName.shadow, RobotName.inspire] #, RobotName.shadow, RobotName.inspire] #[RobotName.shadow, RobotName.allegro]  
    hand_type = [HandType.right] #, HandType.right, HandType.right]
    robot_type = ["inspire"] #["leaphand", "shadowhand", "inspire"] #, "shadowhand", "inspire"] #["shadow", "leap", "inspire"]
    robot_hand_models = [create_hand_model('inspire', torch.device('cpu'))] #[create_hand_model('leaphand', torch.device('cpu'))] #[create_hand_model('leaphand', torch.device('cpu')), create_hand_model('shadowhand', torch.device('cpu')), create_hand_model('inspire', torch.device('cpu'))] #  , create_hand_model('shadowhand', torch.device('cpu')), create_hand_model('inspire', torch.device('cpu'))] #[create_hand_model('shadowhand', torch.device('cpu')), create_hand_model('leaphand', torch.device('cpu')), create_hand_model('inspire', torch.device('cpu'))] #[create_hand_model('shadowhand', torch.device('cpu')), create_hand_model('allegro', torch.device('cpu'))]

    # LEAPHAND: power_drill, microphone, wine_glass | pen (may recheck)
    # Ensure objects is always a list (nargs='+' returns a list, but default might be a string)
    if isinstance(retarget_args.objects, list):
        objects = retarget_args.objects
    else:
        objects = [retarget_args.objects]
    #['bottle_lift0', 'bottle_lift1', 'bottle_lift2', 'bottle_lift3', 'bottle_lift4', 'bottle_lift5', 'bottle_lift6', 'bottle_lift7', 'bottle_lift8', 'bottle_lift9', 'bottle_lift10', 'bottle_lift11'] #['book_pick1', 'book_pick2', 'book_pick3', 'book_pick4', 'book_pick5'] #['g1_plate_grasp11']#['real_plate_3'] #['pick_bowl'] # #['mug', 'tong', 'phone', 'pen'] #['microphone'] #['knife'] #, 'microphone']
    obj_idx = [] #torch.load("<dex-retargeting>/example/position_retargeting/obj_idxs/power_drill_leap.pt") #[] #[21, 29, 66, 68, 74, 112, 116, 118]
    current_dir = os.getcwd()
    # Per-object working dirs. Set DATA_ROOT (process_videos.sh exports it) to
    # point somewhere else; otherwise fall back to <repo>/reconstruction/data.
    # parents[4] is reconstruction/ whether this file is run from added/ or from
    # the copy inside the dex-retargeting submodule.
    BASE_DIR = os.environ.get(
        "DATA_ROOT", str(Path(__file__).resolve().parents[4] / "data")
    )
    print(f"[retarget] data root: {BASE_DIR}")
    use_optimize = retarget_args.use_optimize
    rolling_window = 5
    rolling_qpos_all = []


    # Optional reference results used only for debug printouts / paper comparisons.
    # These .npy files are not shipped with the repo (they are large and specific to
    # our runs), so their absence must not stop a retargeting run.
    comparsion_dir = os.environ.get(
        "RESULT_COMPARISON_DIR",
        str(Path(__file__).resolve().parent / "result_comparison"),
    )
    scale_hand_joints = scale_hand_poses = scale_robot_qpos = None
    try:
        scale_hand_joints = np.load(os.path.join(comparsion_dir, "hand_joints_3d.npy"), allow_pickle=True).item()
        scale_hand_poses = np.load(os.path.join(comparsion_dir, "hand_poses.npy"), allow_pickle=True).item()
        scale_robot_qpos = np.load(os.path.join(comparsion_dir, "robot_qpos.npy"), allow_pickle=True).item()
        print(f"[retarget] loaded reference comparison data from {comparsion_dir}")
    except FileNotFoundError:
        print(f"[retarget] no reference comparison data in {comparsion_dir} "
              f"(optional; set RESULT_COMPARISON_DIR to use it) - continuing without it")

    for robot_idx, hand_type_i, robot_name_i, robot_type_i in zip(range(len(robot_names)), hand_type, robot_names, robot_type):
        for object_name in objects:
            count = 0
            object_mesh_count = 0  
            video_metadata = [] 
            robot_qpos_all = []
            index_all = []
            rgb_dir = os.path.join(BASE_DIR, object_name, "rgb")
            if (use_optimize):
                hand_params_dir = os.path.join(BASE_DIR, object_name, "optimized_hand_object")
            else:
                hand_params_dir = os.path.join(BASE_DIR, object_name, "human_hand")
            original_hand_params_dir = os.path.join(BASE_DIR, object_name, "human_hand")
            object_dir = os.path.join(BASE_DIR, object_name, "obj_mesh")
            # list all the png files in the hand_params_dir and only keep the file names
            folders = sorted([f.split("/")[-1].split(".")[0] for f in glob.glob(os.path.join(rgb_dir, "*.png"))])
            #sorted([f.split("/")[-1].split(".")[0][:-4] for f in glob.glob(os.path.join(original_hand_params_dir, "*.jpg"))])[index:]
            # pdb.set_trace()
            for i, rgb_name in enumerate(folders): #len(folders)): #len(folders)
                hand_params_file = os.path.join(hand_params_dir, f"{rgb_name}_mano_data.npy")
                optimized_hand_params_file = os.path.join(hand_params_dir, f"{rgb_name}_optimized_hand_params.npy")
                if (not os.path.exists(hand_params_file) and not os.path.exists(optimized_hand_params_file)):
                    continue
                try:
                    rgb_name = rgb_name.split(".")[0]
                    object_mesh_path = os.path.join(object_dir, f"{rgb_name}_object_mesh_grasp.obj")
                    if (not os.path.exists(object_mesh_path)):
                        video_object_mesh = None
                    else:
                        video_object_mesh = trimesh.load(object_mesh_path)
                    if (use_optimize):
                        hand_params_file = os.path.join(hand_params_dir, f"{rgb_name}_optimized_hand_params.npy")
                        unoptimized_hand_params_dir = os.path.join(BASE_DIR, object_name, "human_hand")
                        hand_mesh_paths = sorted(glob.glob(os.path.join(unoptimized_hand_params_dir, f"{rgb_name}_right_3d_*.obj")))
                        video_hand_meshs = [trimesh.load(hand_mesh_path, force='mesh') for hand_mesh_path in hand_mesh_paths]
                        video_hand_mesh = video_hand_meshs[0]
                    else:
                        hand_params_file = os.path.join(hand_params_dir, f"{rgb_name}_mano_data.npy")
                        hand_mesh_paths = sorted(glob.glob(os.path.join(hand_params_dir, f"{rgb_name}_right_3d_*.obj")))
                        hand_meshes = [trimesh.load(hand_mesh_path, force='mesh') for hand_mesh_path in hand_mesh_paths]
                        if (video_object_mesh is not None):
                            closest_person_id, min_avg_distance = get_closest_hand(hand_meshes, video_object_mesh)
                        else:
                            closest_person_id = 0
                        video_hand_mesh = hand_meshes[closest_person_id]
                    
                    hand_params = np.load(hand_params_file, allow_pickle=True).item()
                    video_hand_data = {}
                    if (use_optimize):
                        video_hand_data['hand_pose'] = hand_params['hand_pose']
                        video_hand_data['hand_joint'] = hand_params['hand_joint']
                    else:
                        video_hand_data['hand_pose'] = hand_params[f'person_{closest_person_id}']['mano_retarget_hand_pose']
                        video_hand_data['hand_joint'] = hand_params[f'person_{closest_person_id}']['mano_retarget_hand_joints']
                    # Get retarget robot hand qpos with icp transformation matrix 
                except Exception as e:
                    print(f"An unexpected error occurred: {e} in {rgb_name}")
                    continue
                    
                #pdb.set_trace()
                # video_hand_data['hand_joint'] = scale_hand_joints[0][0]
                # video_hand_data['hand_pose'] = scale_hand_poses[0][0]
                robot_qpos = my_retarget_dataset(
                    input_metadata=video_hand_data,
                    robot_names=[robot_name_i],
                    hand_type=[hand_type_i],
                    headless=True, 
                    use_ray_tracing=False,
                    retarget_only=True
                )
                if (robot_qpos is None):
                    continue
                # robot_qpos[robot_names[0]][0][0:6] = [0,0,0,0,0,0]
                # robot_qpos[robot_names[0]][0][5] += np.pi / 2
                # robot_qpos[robot_names[0]][0][3] = 0
                # robot_qpos[robot_names[0]][0][4] = 0
                # robot_qpos[robot_names[0]][0][5] = 0
                # robot_qpos[robot_names[0]][0][0] += 0.3
                # robot_qpos[robot_names[0]][0][2] -= 0.3
                qpos = torch.from_numpy(robot_qpos[robot_name_i][0]).to(torch.float32)
                if scale_robot_qpos is not None and count in scale_robot_qpos:
                    print(f"i {i}, count: {count}, qpos {qpos}, scale_robot_qpos {scale_robot_qpos[count]}")
                else:
                    print(f"i {i}, count: {count}, qpos {qpos}")
                # robot_hand_mesh = robot_hand_models[robot_idx].get_trimesh_q(qpos)['visual']
                #pdb.set_trace()

                # if (len(robot_qpos_all) > rolling_window and i > 10):
                #     #pdb.set_trace()
                #     tmp_robot_qpos_window = np.array(robot_qpos_all)[-rolling_window:-1, :6]
                #     rolling_qpos_previous_mean = np.mean(tmp_robot_qpos_window, axis=0)
                #     # # pdb.set_trace()
                #     # if np.linalg.norm(qpos[:3].numpy() - rolling_qpos_previous_mean[:3]) > 0.06:
                #     #     print(f"i {i}, exceed position rolling window threshold, reset robot qpos from {qpos[:3].numpy()} to {rolling_qpos_previous_mean[:3]}")
                #     #     qpos[:3] = torch.from_numpy(rolling_qpos_previous_mean[:3]).to(torch.float32)
                #     if np.linalg.norm(qpos[4].numpy() - rolling_qpos_previous_mean[4]) > 0.8:
                #         print(f"i {i}, exceed orientation rolling window threshold, reset robot qpos from {qpos[3:6].numpy()} to {rolling_qpos_previous_mean[3:6]}")
                #         #qpos[3:6] = torch.from_numpy(rolling_qpos_previous_mean[3:6]).to(torch.float32)
                #         qpos[4:5] = torch.from_numpy(rolling_qpos_previous_mean[4:5]).to(torch.float32)
                #     if (qpos[5] > 0):
                #         qpos[5:6] = torch.from_numpy(rolling_qpos_previous_mean[5:6]).to(torch.float32)
                qpos = qpos.numpy()
                robot_qpos_all.append(qpos)
                index_all.append(i)    
                # print(f"i {i}, robot_qpos: {qpos[:6]},")
                count += 1
                # if (len(robot_qpos_all) > 50):
                #     break
                # video_hand_mesh, 
                # vis_mesh([robot_hand_mesh, video_hand_mesh, video_object_mesh], obj=object_name, count=i, robot_name=robot_type_i)
            # process_obj_onelink(Path(f"<data_root>/{object_name}/mesh/textured_simple.obj"), args.coacd_args, urdf_args)

            # pdb.set_trace()
            # save the robot_qpos    
            os.makedirs(os.path.join(BASE_DIR, object_name, "robot_qpos", robot_type_i), exist_ok=True)
            np.save(os.path.join(BASE_DIR, object_name, "robot_qpos", robot_type_i, "robot_qpos_index.npy"), index_all)
            #print(f"length of robot_qpos_all: {len(robot_qpos_all)}, path: {f'<data_root>/{object_name}/robot_qpos/{robot_type_i}/robot_qpos_index.npy'}")
            if (use_optimize):
                np.save(os.path.join(BASE_DIR, object_name, "robot_qpos", robot_type_i, "robot_qpos_optimized.npy"), robot_qpos_all)
            else:   
                #print(f"length of robot_qpos_all: {len(robot_qpos_all)}, path: {f'<data_root>/{object_name}/robot_qpos/{robot_type_i}/robot_qpos.npy'}")
                np.save(os.path.join(BASE_DIR, object_name, "robot_qpos", robot_type_i, "robot_qpos.npy"), robot_qpos_all)