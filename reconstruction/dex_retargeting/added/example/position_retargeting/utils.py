import trimesh 
from scipy.stats import wasserstein_distance
import random
import torch 
import numpy as np
import os
from typing import List, Optional, Dict
from trimesh.version import __version__ as trimesh_version
import trimesh as tm
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
import pdb 
import matplotlib.pyplot as plt

def load_dataset(metadata_path: str) -> Dict:
    data = torch.load(metadata_path, map_location='cpu')
    return data


def save_dataset(data: Dict, save_path: str):
    torch.save(data, save_path)
    print(f"Saved dataset to {save_path}")

def vis_mesh(meshs, obj=None, count = 0, robot_name = "", is_video=False):
    scene = trimesh.Scene()
    for mesh in meshs:
        scene.add_geometry(mesh)
    scene.show() 
    # folder_name = f"{obj}_{robot_name}"

    
    # png_data = scene.save_image(resolution=(800, 600), visible=True)  # resolution is optional
    # folder_name = f"{obj}_{robot_name}"
    # os.makedirs(f'./tmp_img/{folder_name}', exist_ok=True)
    # mesh_name = f"mesh_{count}.obj"
    # # scene.export(f'./tmp_img/{folder_name}/{mesh_name}')
    # data = scene.save_image(resolution=(1080,1080))
    # file_name = f"scene_render_{count}_video.png" if is_video else f"scene_render_{count}.png"
    
    # with open(f'./tmp_img/{folder_name}/{file_name}', 'wb') as f:
    #     f.write(png_data)



def vis_pcs(pcs):
    # Create 3D figure
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
    # plt.show(block=False)

    # # # Wait for console input
    # input("Press Enter to close the figure...")

    # # Now close the figure programmatically
    # plt.close(fig)
    # plt.savefig("vis_pcs.png", dpi=300)

def compute_d2_distribution(point_cloud, num_samples=10000, num_bins=50):
    """
    Compute the D2 shape distribution for a point cloud.
    
    Args:
    - point_cloud: (N, 3) numpy array of points.
    - num_samples: Number of point pairs to sample.
    - num_bins: Number of bins in the histogram.
    
    Returns:
    - histogram: Normalized histogram of D2 distances.
    """
    n_points = point_cloud.shape[0]

    # Randomly select point pairs
    indices = np.random.randint(0, n_points, size=(num_samples, 2))
    
    # Compute distances between the point pairs
    distances = np.linalg.norm(point_cloud[indices[:, 0]] - point_cloud[indices[:, 1]], axis=1)
    
    # Compute histogram
    hist, bin_edges = np.histogram(distances, bins=num_bins, density=True)
    
    return hist, bin_edges


def normalize_pcs(pcs, scale = 0.2):
    """
    Normalize a point cloud to fit within a bounding box of size 0.2.

    Parameters:
    pcs (numpy.ndarray): Nx3 array representing the point cloud.

    Returns:
    numpy.ndarray: Scaled and centered point cloud.
    """
    # Compute bounding box
    min_coords = np.min(pcs, axis=0)
    max_coords = np.max(pcs, axis=0)

    # Compute centroid and max extent
    centroid = (min_coords + max_coords) / 2
    max_extent = np.max(max_coords - min_coords)  # Largest dimension

    # Compute scale factor to fit in [-0.1, 0.1]
    scale_factor = scale / max_extent

    # Center the point cloud and apply scaling
    pcs_normalized = (pcs - centroid) * scale_factor

    return pcs_normalized


def normalize_mesh(mesh):
    # Get the bounding box extents
    bounding_box = mesh.bounds
    min_coords, max_coords = bounding_box[0], bounding_box[1]
    max_extent = max(max_coords - min_coords)
    # pdb.set_trace()
    # Compute scale factor to fit in [-0.1, 0.1]
    scale_factor = 0.2 / max_extent

    # Center the object
    centroid = (min_coords + max_coords) / 2
    mesh.apply_translation(-centroid)

    # Scale the object
    mesh.apply_scale(scale_factor)
    return mesh 


def save_mesh_pcs(mesh, output_pc_path, num_points=512, index=None):

    if index is not None:
        object_pc = mesh.vertices[index]
        face_normals = mesh.face_normals[index]
    else:
        object_pc, face_indices = mesh.sample(65536, return_index=True)
        indices = torch.randperm(65536)[:num_points]
        object_pc = object_pc[indices]
        face_normals = mesh.face_normals[face_indices[indices]]
    object_pc = torch.tensor(object_pc, dtype=torch.float32)

    # Compute normals based on the sampled face indices
    
    # Convert normals to torch tensor (float32)
    object_normals = torch.tensor(face_normals, dtype=torch.float32)
    object_pc = torch.cat([object_pc, object_normals], axis=1)
    #pdb.set_trace()
    torch.save(object_pc, output_pc_path)



import os as _os
# Optional YCB-style object model dir; only used by the commented-out YCBArgs
# demos below. Override with YCB_MODELS_DIR if you use them.
YCB_MODELS_DIR_PATH_STR = _os.environ.get("YCB_MODELS_DIR", "")

@dataclass(frozen=True)
class CoacdArgs:
    """Arguments to pass to CoACD.
    Defaults and descriptions are copied from: https://github.com/SarahWeiii/CoACD
    """

    preprocess_resolution: int = 50
    """resolution for manifold preprocess (20~100), default = 50"""
    threshold: float = 0.05
    """concavity threshold for terminating the decomposition (0.01~1), default = 0.05"""
    max_convex_hull: int = -1
    """max # convex hulls in the result, -1 for no maximum limitation"""
    mcts_iterations: int = 100
    """number of search iterations in MCTS (60~2000), default = 100"""
    mcts_max_depth: int = 3
    """max search depth in MCTS (2~7), default = 3"""
    mcts_nodes: int = 20
    """max number of child nodes in MCTS (10~40), default = 20"""
    resolution: int = 2000
    """sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000"""
    pca: bool = False
    """enable PCA pre-processing, default = false"""
    seed: int = 0
    """random seed used for sampling, default = 0"""

@dataclass(frozen=True)
class YCBArgs:
    # obj_dir: str = YCB_MODELS_DIR_PATH_STR
    """path to a directory containing obj files. All obj files in the directory will be converted"""
    coacd_args: CoacdArgs = field(default_factory=CoacdArgs)
    """arguments to pass to CoACD"""
    num_processes: int = 1
    """number of processes to use for multiprocessing"""


@dataclass(frozen=True)
class URDFArgs:
    scale: float = 1.0
    """scale factor for the mesh"""
    density: float = 1000
    """density of the mesh"""

def process_obj_onelink(
    obj_file_path: Path, args: CoacdArgs, urdf_args: URDFArgs, object_type = "grasp", object_idx = None
) -> None:
    """
    - args:
        - obj_file_path: Path

    ---
    since isaacgym now support treating submeshes in the mesh as the convex decomposition of the mesh,
    we can create a URDF file that has only one link, which is the original mesh, and use the submeshes
    as the convex decomposition of the mesh.

    """
    import coacd
    import lxml.etree as et
    import pdb 

    mesh_tm = tm.load(obj_file_path.resolve(), force="mesh")
    mesh = coacd.Mesh(mesh_tm.vertices, mesh_tm.faces)  # type: ignore

    parts = coacd.run_coacd(
        mesh=mesh,
        **asdict(args),
    )
    convex_pieces = []
    for vs, fs in parts:
        convex_pieces.append(tm.Trimesh(vs, fs))
    if (object_idx is not None):
        one_mesh_path = obj_file_path.parent.resolve() / f"coacd_allinone_{object_type}_{object_idx}.obj"
    else:
        one_mesh_path = obj_file_path.parent.resolve() / f"coacd_allinone_{object_type}.obj"
    one_mesh = tm.Scene(convex_pieces)
    one_mesh.export(one_mesh_path)

    scale = urdf_args.scale
    density = urdf_args.density

    if (object_idx is not None):
        urdf_name = f"coacd_decomposed_object_one_link_{object_type}_{object_idx}"
    else:
        urdf_name = f"coacd_decomposed_object_one_link_{object_type}"
    root = et.Element("robot", name="root")

    # create a only visual link using the original mesh
    piece_name = "original"
    link_name = "link_{}".format(piece_name)
    link = et.SubElement(root, "link", name=link_name)
    # Visual Information
    visual = et.SubElement(link, "visual")
    et.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = et.SubElement(visual, "geometry")
    et.SubElement(
        geometry,
        "mesh",
        filename=os.path.basename(str(obj_file_path.resolve())),
        scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale),
    )
    # Inertial information
    mesh_tm.density = density
    I = [["{:.2E}".format(y) for y in x] for x in mesh_tm.moment_inertia]  # NOQA
    inertial = et.SubElement(link, "inertial")
    et.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    et.SubElement(
        inertial,
        "inertia",
        ixx=I[0][0],
        ixy=I[0][1],
        ixz=I[0][2],
        iyy=I[1][1],
        iyz=I[1][2],
        izz=I[2][2],
    )
    # Collision Information
    collision = et.SubElement(link, "collision")
    et.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = et.SubElement(collision, "geometry")
    if (object_idx is not None):
        filename = f"coacd_allinone_{object_type}_{object_idx}.obj"
    else:
        filename = f"coacd_allinone_{object_type}.obj"
    et.SubElement(
        geometry,
        "mesh",
        filename=filename,
        scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale),
    )
    # Write the link out to the XML Tree
    link = et.SubElement(root, "link", name=link_name)

    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = "{}.urdf".format(urdf_name)
    tree.write(
        os.path.join(obj_file_path.resolve().parent.as_posix(), urdf_filename),
        pretty_print=True,
    )
    return np.sum(convex_pieces)

def get_closest_hand(hand_meshes, object_mesh):
    closest_person_id = None
    closest_hand_mesh = None
    min_avg_distance = float('inf')

    for person_id, hand_mesh in enumerate(hand_meshes):
        # Calculate average distance from hand to object center
        hand_center_3d = np.mean(hand_mesh.vertices, axis=0)
        obj_center_3d = np.mean(object_mesh.vertices, axis=0)
        avg_distance = np.linalg.norm(hand_center_3d - obj_center_3d)
        
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            closest_person_id = person_id
            closest_hand_mesh = hand_mesh
    return closest_person_id, min_avg_distance

def apply_rotation_around_origin_to_hand_data(hand_joints, rotation_matrix):
    """
    Apply rotation transformation around origin to hand joints and mesh
    
    Args:
        hand_joints: Hand joint positions (N, 3)
        hand_mesh: Hand mesh object
        rotation_matrix: 4x4 rotation matrix around origin
    
    Returns:
        transformed_hand_joints: Rotated joint positions
        transformed_hand_mesh: Rotated mesh
    """
    # Transform hand joints around origin
    ones = np.ones((hand_joints.shape[0], 1))
    hand_joints_homogeneous = np.hstack([hand_joints, ones])
    transformed_hand_joints = np.dot(hand_joints_homogeneous, rotation_matrix.T)[:, :3]

    
    return transformed_hand_joints