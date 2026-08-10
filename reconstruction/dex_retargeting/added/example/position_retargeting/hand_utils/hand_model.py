import os
import sys
import json
import math
import random
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk

from urdfpy import URDF
import fpsample

# Robot URDFs, link meshes and per-robot point clouds live in the DRO-Grasp
# repository, which is a separate checkout (see reconstruction/README.md).
# Point DRO_GRASP_ROOT at it; the default is a sibling checkout next to VideoManip.
ROOT_DIR = os.environ.get(
    "DRO_GRASP_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 7), "DRO-Grasp"),
)
ROOT_DIR = os.path.normpath(ROOT_DIR)
if not os.path.isdir(ROOT_DIR):
    raise FileNotFoundError(
        f"DRO-Grasp assets not found at {ROOT_DIR}. The hand retargeting stage needs the "
        f"DRO-Grasp repository for robot URDFs and point clouds. Clone it and set "
        f"DRO_GRASP_ROOT=/path/to/DRO-Grasp (see reconstruction/README.md)."
    )
sys.path.append(ROOT_DIR)

from hand_utils.func_utils import farthest_point_sampling
from hand_utils.mesh_utils import load_link_geometries
from hand_utils.rotation import *
import pdb 



class HandModel:
    def __init__(
        self,
        robot_name,
        urdf_path,
        meshes_path,
        links_pc_path,
        device,
        link_num_points=512
    ):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.meshes_path = meshes_path
        self.device = device
        
        self.pk_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        if os.path.exists(links_pc_path):  # In case of generating robot links pc, the file doesn't exist.
            links_pc_data = torch.load(links_pc_path, map_location=device, weights_only=False)
            self.links_pc = links_pc_data['filtered']
            self.links_pc_original = links_pc_data['original']
        else:
            self.links_pc = None
            self.links_pc_original = None
        self.meshes = load_link_geometries(robot_name, self.urdf_path, self.pk_chain.get_link_names())

        self.vertices = {}
        removed_json_data = json.load(open(os.path.join(ROOT_DIR, 'data_utils/removed_links_new.json')))
        removed_links = removed_json_data[robot_name]
        # removed_links = []
        for link_name, link_mesh in self.meshes.items():
            if link_name in removed_links:  # remove links unrelated to contact
                continue
            v = link_mesh.sample(link_num_points)
            self.vertices[link_name] = v
        self.frame_status = None


    def load_robot_links_point_clouds(self, urdf_file_path, num_samples=512):
        """
        Load point clouds for each link in a robot URDF file.

        Parameters:
            urdf_file (str): Path to the URDF file.

        Returns:
            dict: A dictionary mapping link names to their point clouds.
        """
        robot = URDF.load(urdf_file_path)
        link_point_clouds = {}
        for link in robot.links:
            if link.visuals:
                for visual in link.visuals:
                    if visual.geometry.mesh:
                        mesh_path = visual.geometry.mesh.filename
                        # path = os.path.join("urdf", mesh_path)
                        # print(f"mesh path {mesh_path}")
                        pdb.set_trace()
                        mesh = trimesh.load_mesh(os.path.join("mano-urdf/urdf", mesh_path))
                        sampled_points = trimesh.sample.sample_surface(mesh, num_samples)[0]
                        if link.name not in link_point_clouds:
                            link_point_clouds[link.name] = sampled_points
                        else:
                            link_point_clouds[link.name] = np.vstack((
                                link_point_clouds[link.name], sampled_points
                            ))
                        # print(f"link.name {link.name}")

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
        return {
            "original": link_point_clouds,
            "filtered": fps_points_by_link
        }


    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_transformed_links_pc(self, q=None, links_pc=None):
        """
        Use robot link pc & q value to get point cloud.

        :param q: (6 + DOF,), joint values (euler representation)
        :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
        :return: point cloud: (N, 4), with link index
        """
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc

        all_pc_se3 = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]
            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
            index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
            all_pc_se3.append(link_pc_se3_index)
            # print(f"link_name {link_name}, len(all_pc_se3) {len(link_pc_se3_index)}, len(all_pc_se3) {len(torch.cat(all_pc_se3, dim=0))}")
        all_pc_se3 = torch.cat(all_pc_se3, dim=0)
        return all_pc_se3




    def get_sampled_pc(self, q=None, num_points=512):
        """
        :param q: (9 + DOF,), joint values (rot6d representation)
        :param num_points: int, number of sampled points
        :return: ((N, 3), list), sampled point cloud (numpy) & index
        """
        if q is None:
            q = self.get_canonical_q()
        # pdb.set_trace()
        # sampled_pc = self.get_transformed_links_pc(q, self.vertices)
        # pdb.set_trace()
        # fps_samples_idx = fpsample.fps_sampling(sampled_pc[:, :3], num_points)
        # pdb.set_trace()
        # return sampled_pc[fps_samples_idx, :3]

        sampled_pc = self.get_transformed_links_pc(q) #, self.vertices
        return sampled_pc[:, :3]
        # return farthest_point_sampling(sampled_pc, num_points)

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.pk_chain.get_joint_limits()
        canonical_q = torch.tensor(lower) * 0.75 + torch.tensor(upper) * 0.25
        canonical_q[:6] = 0
        return canonical_q

    def get_initial_q(self, q=None, max_angle: float = math.pi / 6):
        """
        Compute the robot initial joint value q based on the target grasp.
        Root translation is not considered since the point cloud will be normalized to zero-mean.

        :param q: (6 + DOF,) or (9 + DOF,), joint values (euler/rot6d representation)
        :param max_angle: float, maximum angle of the random rotation
        :return: initial q: (6 + DOF,), euler representation
        """
        if q is None:  # random sample root rotation and joint values
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

            q_initial[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
            q_initial[5] /= 2

            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[6:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
        else:
            if len(q) == self.dof:
                q = q_euler_to_q_rot6d(q)
            q_initial = q.clone()

            # compute random initial rotation
            direction = - q_initial[:3] / torch.norm(q_initial[:3])
            angle = torch.tensor(random.uniform(0, max_angle), device=q.device)  # sample rotation angle
            axis = torch.randn(3).to(q.device)  # sample rotation axis
            axis -= torch.dot(axis, direction) * direction  # ensure orthogonality
            axis = axis / torch.norm(axis)
            random_rotation = axisangle_to_matrix(axis, angle).to(q.device)
            rotation_matrix = random_rotation @ rot6d_to_matrix(q_initial[3:9])
            q_initial[3:9] = matrix_to_rot6d(rotation_matrix)

            # compute random initial joint values
            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[9:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
            # q_initial[9:] = torch.zeros_like(q_initial[9:], dtype=q.dtype, device=q.device)

            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial


    def get_trimesh_q(self, q):
        """ Return the hand trimesh object corresponding to the input joint value q. """
        self.update_status(q)
        vertices = []
        faces = []
        vertex_offset = 0
        scene = trimesh.Scene()
        for link_name in self.vertices:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            # scene.add_geometry(mesh)
            # scene.add_geometry(self.meshes[link_name].copy())
            # 
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(mesh.vertices)
            # pdb.set_trace()

        # pdb.set_trace() 
        # for geom in scene.geometry.values():
        #     if isinstance(geom, trimesh.Trimesh):
        #         vertices.append(geom.vertices)
        #         faces.append(geom.faces + vertex_offset)
        #         vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        parts = {}
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            parts[link_name] = part_mesh

        return_dict = {
            'visual': trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        """ Return the hand trimesh object corresponding to the input transform. """
        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    def set_qpos(self, q):
        """
        Set the joint positions of the hand model.
        
        Args:
            q: torch.Tensor, joint positions (can be in euler or rot6d format)
        """
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.update_status(q)

    def get_link_pose(self, link_name):
        """
        Get the pose (4x4 transformation matrix) of a specific link.
        
        Args:
            link_name: str, name of the link
            
        Returns:
            torch.Tensor: 4x4 transformation matrix for the link
        """
        if self.frame_status is None:
            # Initialize with zero joint positions if not set
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
            self.update_status(q)
        
        if link_name not in self.frame_status:
            raise ValueError(f"Link '{link_name}' not found in the robot model")
        return self.frame_status[link_name].get_matrix()[0] 


def create_hand_model(
    robot_name,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_points=512
):
    json_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/urdf_assets_meta_new.json') #urdf_assets_meta.json
    urdf_assets_meta = json.load(open(json_path))
    urdf_path = os.path.join(ROOT_DIR, urdf_assets_meta['urdf_path'][robot_name])
    meshes_path = os.path.join(ROOT_DIR, urdf_assets_meta['meshes_path'][robot_name])
    links_pc_path = os.path.join(ROOT_DIR, f'data/PointCloud/robot/{robot_name}.pt')
    # current_dir = os.getcwd()
    # urdf_path = os.path.join(current_dir, robot_name, 'inspire_hand_right_extended.urdf')
    # meshes_path = os.path.join(current_dir, robot_name, 'meshes')
    # links_pc_path = os.path.join(current_dir, robot_name, f'{robot_name}.pt')
    hand_model = HandModel(robot_name, urdf_path, meshes_path, links_pc_path, device, num_points)
    return hand_model
