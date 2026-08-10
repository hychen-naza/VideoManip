from pathlib import Path
from typing import List, Optional

import cv2
from tqdm import trange
import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from sapien.asset import create_dome_envmap
from pytransform3d import transformations as pt


def compute_smooth_shading_normal_np(vertices, indices):
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1) 

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


class RenderBase:
    def __init__(self, headless: bool = False, use_ray_tracing: bool = False):
        if not use_ray_tracing:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        else:
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")

        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)

        self.scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        self.scene.add_directional_light(np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
        self.scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        self.scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        self.scene.add_ground(-1, render_material=visual_material)

        if not headless:
            self.viewer = Viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(1.5, 0, 1)
            self.viewer.set_camera_rpy(0, -0.8, 3.14)
            self.viewer.control_window.toggle_origin_frame(False)
        else:
            self.camera = self.scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
            self.camera.set_local_pose(sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061]))

        self.headless = headless

        self._add_table()

        self.internal_scene = self.scene.render_system._internal_scene
        self.context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)

        self.objects: List[sapien.Entity] = []
        self.nodes: List[sapien.render.Node] = []

    def _add_table(self):
        white_diffuse = sapien.render.RenderMaterial()
        white_diffuse.set_base_color(np.array([0.8, 0.8, 0.8, 1]))
        white_diffuse.set_roughness(0.9)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02]))
        builder.add_box_visual(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02]), material=white_diffuse)
        edges = [
            sapien.Pose([0.4, 1.9, -0.51]),
            sapien.Pose([-0.4, 1.9, -0.51]),
            sapien.Pose([0.4, -1.9, -0.51]),
            sapien.Pose([-0.4, -1.9, -0.51])
        ]
        for edge_pose in edges:
            builder.add_box_visual(
                edge_pose, half_size=np.array([0.015, 0.015, 0.49]), material=white_diffuse
            )
        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose([0.5, 0, 0]))

    def clear_all(self):
        for actor in self.objects:
            self.scene.remove_actor(actor)
        self.objects.clear()
        self.clear_node()

    def clear_node(self):
        for node in self.nodes:
            self.internal_scene.remove_node(node)
        self.nodes.clear()

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):
        raise NotImplementedError

    def _update_hand(self, vertex, mano_face):
        self.clear_node()
        normal = compute_smooth_shading_normal_np(vertex, mano_face)
        mesh = self.context.create_mesh_from_array(vertex, mano_face, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)

    def render(self, vertex: np.ndarray, joint: np.ndarray, mano_face: np.ndarray, 
               object_pose: List[np.ndarray], fps: int = 10, video_path: Optional[Path] = None):
        frame_num = len(joint)
        step_per_frame = int(60 / fps)

        writer = None
        if self.headless and video_path is not None:
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (self.camera.get_width(), self.camera.get_height())
            )

        for i in trange(frame_num, desc="Rendering"):
            vertex_frame = vertex[i]
            joint_frame = joint[i]

            if vertex_frame is not None:
                self._update_hand(vertex_frame, mano_face)

            for k, pos_quat in enumerate(object_pose[i]):
                pose = sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
                self.objects[k].set_pose(pose)

            self.scene.update_render()

            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                if writer:
                    writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        if writer:
            writer.release()
            print(f"Video saved to {video_path}")

    def close(self):
        if not self.headless and hasattr(self, 'viewer'):
            self.viewer.close_window()