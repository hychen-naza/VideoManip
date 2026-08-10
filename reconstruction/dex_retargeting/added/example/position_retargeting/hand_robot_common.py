import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sapien.core as sapien
from pytransform3d import rotations
from tqdm import trange

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
import cv2
import torch
import pdb 



class RobotHandRetargeting(RenderBase):
    dro_shadow_hand_joint_names = [
        'dummy_x_translation_joint',
        'dummy_y_translation_joint',
        'dummy_z_translation_joint',
        'dummy_x_rotation_joint',
        'dummy_y_rotation_joint',
        'dummy_z_rotation_joint',
        'WRJ2',
        'WRJ1',
        'FFJ4',
        'FFJ3',
        'FFJ2',
        'FFJ1',
        'MFJ4',
        'MFJ3',
        'MFJ2',
        'MFJ1',
        'RFJ4',
        'RFJ3',
        'RFJ2',
        'RFJ1',
        'LFJ5',
        'LFJ4',
        'LFJ3',
        'LFJ2',
        'LFJ1',
        'THJ5',
        'THJ4',
        'THJ3',
        'THJ2',
        'THJ1'
    ]
    dro_allegro_hand_joint_names = [
        'dummy_x_translation_joint',
        'dummy_y_translation_joint',
        'dummy_z_translation_joint',
        'dummy_x_rotation_joint',
        'dummy_y_rotation_joint',
        'dummy_z_rotation_joint',
        'joint_0.0',
        'joint_1.0',
        'joint_2.0',
        'joint_3.0', 
        'joint_4.0', 
        'joint_5.0', 
        'joint_6.0', 
        'joint_7.0', 
        'joint_8.0', 
        'joint_9.0', 
        'joint_10.0', 
        'joint_11.0', 
        'joint_12.0', 
        'joint_13.0', 
        'joint_14.0', 
        'joint_15.0'
    ]
    dro_leap_hand_joint_names = [
        'dummy_x_translation_joint',
        'dummy_y_translation_joint',
        'dummy_z_translation_joint',
        'dummy_x_rotation_joint',
        'dummy_y_rotation_joint',
        'dummy_z_rotation_joint',
        '1', 
        '0', 
        '2', 
        '3', 
        '5', 
        '4', 
        '6', 
        '7', 
        '9', 
        '8', 
        '10', 
        '11', 
        '12', 
        '13', 
        '14', 
        '15'
    ]
    dro_umi_hand_joint_names = [
        'dummy_x_translation_joint',
        'dummy_y_translation_joint',
        'dummy_z_translation_joint',
        'dummy_x_rotation_joint',
        'dummy_y_rotation_joint',
        'dummy_z_rotation_joint',
        'left_finger_joint',
        'right_finger_joint'
    ]
    dro_panda_hand_joint_names = [
        'dummy_x_translation_joint',
        'dummy_y_translation_joint',
        'dummy_z_translation_joint',
        'dummy_x_rotation_joint',
        'dummy_y_rotation_joint',
        'dummy_z_rotation_joint',
        'panda_finger_joint1',
        'panda_finger_joint2'
    ]
    dro_inspire_hand_joint_names = [
        'dummy_x_translation_joint',
        'dummy_y_translation_joint',
        'dummy_z_translation_joint',
        'dummy_x_rotation_joint',
        'dummy_y_rotation_joint',
        'dummy_z_rotation_joint',
        'thumb_proximal_yaw_joint',
        'thumb_proximal_pitch_joint',
        'thumb_intermediate_joint',
        'thumb_distal_joint',
        'index_proximal_joint',
        'index_intermediate_joint',
        'middle_proximal_joint',
        'middle_intermediate_joint',
        'ring_proximal_joint',
        'ring_intermediate_joint',
        'pinky_proximal_joint',
        'pinky_intermediate_joint'
    ]
    dro_joint_names = {RobotName.shadow: dro_shadow_hand_joint_names, RobotName.leap: dro_leap_hand_joint_names, RobotName.allegro: dro_allegro_hand_joint_names, RobotName.umi: dro_umi_hand_joint_names, RobotName.panda: dro_panda_hand_joint_names, RobotName.inspire: dro_inspire_hand_joint_names}

    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType = "right",
        headless: bool = False,
        use_ray_tracing: bool = False,
        retarget_only: bool = False
    ):
        super().__init__(headless, use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []

        self.retarget2sapien: List[np.ndarray] = []
        self.retarget2sdro: List[np.ndarray] = []
        self.hand_type = hand_type
        self.retarget_only = retarget_only

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        for i, (robot_name, robot_hand_type) in enumerate(zip(robot_names, hand_type)):
            config_path = get_default_config_path(robot_name, RetargetingType.position, robot_hand_type)

            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            # pdb.set_trace()
            retargeting = config.build()
            
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            urdf_path = Path(config.urdf_path)
            # print(f"Loading URDF file: {urdf_path}")
            # if "glb" not in urdf_path.stem:
                # urdf_path = urdf_path.with_name(urdf_path.stem + "_glb.urdf")

            robot_urdf = urdf.URDF.load(
                str(urdf_path),
                add_dummy_free_joints=False,
                build_scene_graph=False
            )
            #pdb.set_trace()
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_path.name}"
            robot_urdf.write_xml_file(temp_path)
            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array([
                retargeting.joint_names.index(n) for n in sapien_joint_names
            ]).astype(int)
            retarget2dro = np.array([
                retargeting.joint_names.index(n) for n in RobotHandRetargeting.dro_joint_names[robot_name]
            ]).astype(int)
            self.retarget2sapien.append(retarget2sapien)
            self.retarget2sdro.append(retarget2dro)


    def retarget_hands(self, rpy, joint):

        robot_qpos = {
            robot_name: [] for robot_name in self.robot_names
        }

        for robot_name, robot, retargeting, retarget2dro, retarget2sapien in zip(
            self.robot_names, self.robots, self.retargetings, self.retarget2sdro, self.retarget2sapien
        ):
            retargeting.reset()  # For one frame retargeting
            wrist_quat = rotations.quaternion_from_compact_axis_angle(rpy[0, 0:3])
            # wrist_quat = rotations.quaternion_from_euler(rpy[0], 0, 1, 2, extrinsic=False)
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=False,
            )

            indices = retargeting.optimizer.target_link_human_indices
            ref_value = joint[indices, :]
            # YUNCHAO: Temoporary qpos wihtout optimization 
            qpos = retargeting.last_qpos
            # Hongyi: Temoporary qpos 
            # pdb.set_trace()
            # qpos = np.concatenate([qpos, [qpos[-1]]])
            # if (robot_name == RobotName.shadow):
            #     # qpos[4] += np.pi #/2
            #     qpos[5] -= np.pi
            # if (robot_name == RobotName.leap):
            #     qpos[5] -= np.pi

            retargeting.last_qpos = qpos
            
            # pdb.set_trace()
            for _ in range(10):
                qpos = retargeting.retarget(ref_value)
            # Debug only
            # if (len(qpos) == 12):
            #     qpos = np.concatenate([qpos[:6], [0.0]*12])
            #pdb.set_trace()
            dro_qos = qpos[retarget2dro]
            robot_qpos[robot_name].append(dro_qos)
            # assert qpos == retargeting.get_qpos(), "Retargeting qpos mismatch"
            
            spain_qos = qpos[retarget2sapien]
            robot.set_qpos(spain_qos)
            retargeting.reset()  # For one frame retargeting

        if not self.retarget_only:
            self.scene.update_render()

        return robot_qpos
    
    def close(self):
        super().close()

    # def render_and_retarget(self, vertex: np.ndarray, joint: np.ndarray, mano_face: np.ndarray, hand_pose: np.ndarray,
    #                         object_pose: List[np.ndarray] = None,  fps: int = 5, y_offset = 0.8, video_path: Optional[Path] = None):

    #     global_y_offset = -y_offset * len(self.robots) / 2
    #     self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
    #     if not self.headless:
    #         self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
    #     else:
    #         local_pose = self.camera.get_local_pose()
    #         local_pose.set_p(np.array([1.5, global_y_offset, 1]))
    #         self.camera.set_local_pose(local_pose)

    #     frame_num = len(joint)

    #     assert frame_num == 1, "Only support one frame retargeting for v1.2.0"

    #     pose_offsets = []
    #     step_per_frame = int(60 / fps)

    #     for i in range(len(self.robots) + 1):
    #         pose = sapien.Pose([0, -y_offset * i, 0])
    #         pose_offsets.append(pose)
    #         if i >= 1:
    #             self.robots[i - 1].set_pose(pose)

    #     writer = None
    #     if self.headless and video_path is not None:
    #         writer = cv2.VideoWriter(
    #             str(video_path),
    #             cv2.VideoWriter_fourcc(*"mp4v"),
    #             1.0,
    #             (self.camera.get_width(), self.camera.get_height())
    #         )

    #     for i in trange(frame_num, desc="Rendering and Retargeting"):
    #         vertex_frame = vertex[i]
    #         joint_frame = joint[i]
    #         hand_pose = hand_pose[i]
    #         mano_face_frame = mano_face

    #         dro_qos = self.retarget_hands(hand_pose, joint_frame)
    #         self._update_hand(vertex_frame, mano_face_frame)

    #         # for k, pos_quat in enumerate(object_pose[i]):
    #         #     pose = sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
    #         #     self.objects[k].set_pose(pose)

    #         self.scene.update_render()

    #         if self.headless:
    #             self.camera.take_picture()
    #             rgb = self.camera.get_picture("Color")[..., :3]
    #             rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    #             if writer:
    #                 writer.write(rgb[..., ::-1])
    #         else:
    #             for _ in range(step_per_frame):
    #                 self.viewer.render()

    #     if not self.headless:
    #         self.viewer.paused = True
    #         self.viewer.render()
    #     if writer:
    #         writer.release()
    #         print(f"Video saved to {video_path}")

    #     return dro_qos

