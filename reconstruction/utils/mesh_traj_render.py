import os
import glob
import trimesh
import numpy as np
import cv2
from PIL import Image
import argparse
import pyrender
import torch
from tqdm import tqdm
import pdb 
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# The retargeting helpers live in the dex-retargeting wrapper; resolve them
# relative to this file so the script works from any checkout.
sys.path.append(str(Path(__file__).resolve().parents[1] /
                    "dex_retargeting" / "dex-retargeting" / "example" / "position_retargeting"))
from hand_utils.hand_model import create_hand_model



def project_meshes_to_2d_video(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir, robot_qpos_dir):
    """
    Project 3D meshes to 2D and save as video
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) #[:10]

    robot_qpos_index_path = os.path.join(robot_qpos_dir, "robot_qpos_index.npy")
    print(f"length of robot_qpos_index_all: {len(np.load(robot_qpos_index_path))}")
    robot_qpos_index_all = np.load(robot_qpos_index_path)

    rgb_files = [rgb_files[i] for i in robot_qpos_index_all]
    # Load camera intrinsics
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    # Setup video writer
    video_path = os.path.join(output_dir, f"projection.mp4")
    # how to check if the video path is valid?
    if not os.path.exists(video_path):
        print(f"Video path does not exist: {video_path}")
        # make the directory
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Get video dimensions from first frame
    first_frame = cv2.imread(rgb_files[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Creating projected 2D video: {video_path}")
    # pdb.set_trace()
    for i, rgb_file in enumerate(tqdm(rgb_files, desc="Projecting frames")):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        
        # Load RGB image
        rgb_image = cv2.imread(rgb_file)
        if rgb_image is None:
            print(f"Could not load image: {rgb_file}")
            continue
        
        # Load hand mesh files
        hand_mesh_pattern = os.path.join(hand_params_dir, f"{image_name}_right_3d_*.obj")
        hand_mesh_files = glob.glob(hand_mesh_pattern)
        
        # Load object mesh
        object_mesh_pattern = os.path.join(object_dir, f"{image_name}_object_mesh_*.obj")
        object_mesh_files = glob.glob(object_mesh_pattern)
        # Project object mesh to 2D
        for object_mesh_file in object_mesh_files:
            try:
                obj_mesh = trimesh.load(object_mesh_file, force='mesh')
                if obj_mesh.vertices.shape[0] > 0:
                    # Project vertices to 2D
                    vertices_2d, _ = cv2.projectPoints(
                        obj_mesh.vertices,
                        np.zeros(3),  # rotation vector (zero since vertices are already in camera space)
                        np.zeros(3),  # translation vector (zero since vertices are already in camera space)
                        K, 
                        None
                    )
                    vertices_2d = vertices_2d.reshape(-1, 2)
                    
                    # Draw object mesh vertices on the image
                    for vertex in vertices_2d:
                        x, y = int(vertex[0]), int(vertex[1])
                        if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
                            cv2.circle(rgb_image, (x, y), 2, (0, 255, 0), -1)  # Green dots for object vertices
            except Exception as e:
                print(f"Error projecting object mesh {object_mesh_path}: {e}")
        
        # Project hand meshes to 2D
        for hand_mesh_file in hand_mesh_files:
            try:
                hand_mesh = trimesh.load(hand_mesh_file, force='mesh')
                if hand_mesh.vertices.shape[0] > 0:
                    # Project vertices to 2D
                    hand_vertices_2d, _ = cv2.projectPoints(
                        hand_mesh.vertices,
                        np.zeros(3),
                        np.zeros(3),
                        K,
                        None
                    )
                    hand_vertices_2d = hand_vertices_2d.reshape(-1, 2)
                    
                    # Draw hand mesh vertices on the image
                    for vertex in hand_vertices_2d:
                        x, y = int(vertex[0]), int(vertex[1])
                        if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
                            cv2.circle(rgb_image, (x, y), 2, (255, 0, 0), -1)  # Blue dots for hand vertices
            except Exception as e:
                print(f"Error projecting hand mesh {hand_mesh_file}: {e}")
        # pdb.set_trace()
        # Write frame to video
        video_writer.write(rgb_image)
    
    # Clean up
    video_writer.release()
    print(f"Projected 2D video saved: {video_path}")


def render_2d_vertices_video_manual(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir):
    """
    Render 2D vertices of hand and object meshes using manual projection and save as video with matplotlib
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) #[:10]
    
    # Load camera intrinsics
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    # Extract camera parameters from K matrix
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point
    
    print(f"Camera intrinsics - fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")
    
    # Setup video path
    video_path = os.path.join(output_dir, f"{os.path.basename(folder_name)}_2d_vertices_manual_matplotlib.mp4")
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Get image dimensions from first frame
    first_frame = cv2.imread(rgb_files[0])
    height, width, layers = first_frame.shape
    
    print(f"Creating 2D vertices video with manual projection using matplotlib: {video_path}")
    
    def manual_project_points(vertices_3d, fx, fy, cx, cy):
        """
        Manually project 3D points to 2D using perspective projection
        """
        # Handle case where Z might be zero or very small
        vertices_3d = np.array(vertices_3d)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        Z = np.maximum(np.abs(vertices_3d[:, 2]), epsilon) * np.sign(vertices_3d[:, 2])
        
        # Manual perspective projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
        u = fx * vertices_3d[:, 0] / Z + cx
        v = fy * vertices_3d[:, 1] / Z + cy
        
        return np.column_stack([u, v])
    
    # Store all frame data for animation
    frame_data = []
    
    for i, rgb_file in enumerate(tqdm(rgb_files, desc="Loading mesh data for manual projection")):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        
        # Load hand mesh files
        hand_mesh_pattern = os.path.join(hand_params_dir, f"{image_name}_right_3d_*.obj")
        hand_mesh_files = glob.glob(hand_mesh_pattern)
        
        # Load object mesh
        object_mesh_path = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
        
        # Collect projected vertices for this frame
        hand_vertices_2d = []
        object_vertices_2d = []
        
        # Project object mesh to 2D using manual projection
        if os.path.exists(object_mesh_path):
            try:
                obj_mesh = trimesh.load(object_mesh_path, force='mesh')
                if obj_mesh.vertices.shape[0] > 0:
                    # Manual projection instead of cv2.projectPoints
                    vertices_2d = manual_project_points(obj_mesh.vertices, fx, fy, cx, cy)
                    object_vertices_2d.append(vertices_2d)
            except Exception as e:
                print(f"Error projecting object mesh {object_mesh_path}: {e}")
        
        # Project hand meshes to 2D using manual projection
        for hand_mesh_file in hand_mesh_files:
            try:
                hand_mesh = trimesh.load(hand_mesh_file, force='mesh')
                if hand_mesh.vertices.shape[0] > 0:
                    # Manual projection instead of cv2.projectPoints
                    vertices_2d = manual_project_points(hand_mesh.vertices, fx, fy, cx, cy)
                    hand_vertices_2d.append(vertices_2d)
            except Exception as e:
                print(f"Error projecting hand mesh {hand_mesh_file}: {e}")
        
        # Store frame data
        frame_data.append({
            'hand_vertices_2d': hand_vertices_2d,
            'object_vertices_2d': object_vertices_2d,
            'frame_name': image_name
        })
    
    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip Y axis to match image coordinates
    ax.set_aspect('equal')
    ax.set_title('2D Hand and Object Vertices (Manual Projection)')
    
    # Create animation
    def animate(frame_idx):
        ax.clear()
        
        # Set axis properties again
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip Y axis to match image coordinates
        ax.set_aspect('equal')
        ax.set_title(f'2D Hand and Object Vertices (Manual Projection) - Frame {frame_idx}')
        
        if frame_idx < len(frame_data):
            frame = frame_data[frame_idx]
            
            # Plot hand vertices in blue
            for hand_verts in frame['hand_vertices_2d']:
                if len(hand_verts) > 0:
                    ax.scatter(hand_verts[:, 0], hand_verts[:, 1], 
                              c='blue', s=20, alpha=0.7, label='Hand' if len(frame['hand_vertices_2d']) == 1 else "")
            
            # Plot object vertices in red
            for obj_verts in frame['object_vertices_2d']:
                if len(obj_verts) > 0:
                    ax.scatter(obj_verts[:, 0], obj_verts[:, 1], 
                              c='red', s=20, alpha=0.7, label='Object' if len(frame['object_vertices_2d']) == 1 else "")
        
        # Add legend only once
        if frame_idx == 0:
            ax.legend()
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(frame_data), 
                                  interval=100, blit=False, repeat=True)
    
    # Save animation as video
    print("Saving animation as video...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(video_path, writer=writer)
    
    plt.close(fig)
    print(f"Manual 2D vertices video saved: {video_path}")


def render_3d_vertices_video(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir, optimized = False):
    """
    Render 3D vertices of hand and object meshes using matplotlib and save as video
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) #[:10]
    
    # Load camera intrinsics
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    print(f"Loaded camera intrinsics K:\n{K}")
    
    # Extract focal lengths and principal point from K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Get image dimensions from first frame
    first_frame = cv2.imread(rgb_files[0])
    height, width = first_frame.shape[:2]
    
    # Setup video writer
    if (optimized):
        video_path = os.path.join(output_dir, f"{os.path.basename(folder_name)}_optimized_3d_vertices_matplotlib.mp4")
    else:
        video_path = os.path.join(output_dir, f"{os.path.basename(folder_name)}_3d_vertices_matplotlib.mp4")
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    print(f"Creating 3D vertices video with matplotlib: {video_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
    print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")
    
    # Calculate scene bounds from first frame
    first_rgb_file = rgb_files[0]
    first_image_name = os.path.basename(first_rgb_file).replace(".png", "")
    
    # Load meshes from first frame to calculate scene bounds
    first_hand_pattern = os.path.join(hand_params_dir, f"{first_image_name}_right_3d_*.obj")
    first_hand_files = glob.glob(first_hand_pattern)
    first_object_pattern = os.path.join(object_dir, f"{first_image_name}_object_mesh_*.obj")
    first_object_files = glob.glob(first_object_pattern)
    
    # Calculate scene bounds from first frame (with transformation applied)
    first_frame_vertices = []
    for hand_mesh_file in first_hand_files:
        try:
            hand_mesh = trimesh.load(hand_mesh_file, force='mesh')
            if hand_mesh.vertices.shape[0] > 0:
                # Apply the same transformation as we'll use in the animation
                from scipy.spatial.transform import Rotation
                vertices = hand_mesh.vertices
                
                # Use the exact transformation matrix: X_new=-Z_old, Y_new=X_old, Z_new=-Y_old
                # Transformation matrix: [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
                transform_matrix = np.array([[0, 0, -1], 
                                            [1, 0, 0], 
                                            [0, -1, 0]])
                rotation = Rotation.from_matrix(transform_matrix)
                vertices_transformed = rotation.apply(vertices)
                
                first_frame_vertices.append(vertices_transformed)
        except Exception as e:
            print(f"Error loading first frame hand mesh {hand_mesh_file}: {e}")
    
    for first_object_file in first_object_files:
        try:
            obj_mesh = trimesh.load(first_object_file, force='mesh')
            if obj_mesh.vertices.shape[0] > 0:
                # Apply the same transformation
                from scipy.spatial.transform import Rotation
                vertices = obj_mesh.vertices
                
                # Use the exact transformation matrix: X_new=-Z_old, Y_new=X_old, Z_new=-Y_old
                # Transformation matrix: [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
                transform_matrix = np.array([[0, 0, -1], 
                                            [1, 0, 0], 
                                            [0, -1, 0]])
                rotation = Rotation.from_matrix(transform_matrix)
                vertices_transformed = rotation.apply(vertices)
                
                first_frame_vertices.append(vertices_transformed)
        except Exception as e:
            print(f"Error loading first frame object mesh {first_object_file}: {e}")
    
    # Calculate scene bounds from transformed vertices
    if first_frame_vertices:
        all_first_vertices = np.vstack(first_frame_vertices)
        scene_center = np.mean(all_first_vertices, axis=0)
        scene_min = np.min(all_first_vertices, axis=0)
        scene_max = np.max(all_first_vertices, axis=0)
        scene_extent = scene_max - scene_min
        max_extent = np.max(scene_extent)
        print(f"Transformed vertices range:")
        print(f"  X: [{scene_min[0]:.3f}, {scene_max[0]:.3f}]")
        print(f"  Y: [{scene_min[1]:.3f}, {scene_max[1]:.3f}]")
        print(f"  Z: [{scene_min[2]:.3f}, {scene_max[2]:.3f}]")
    else:
        scene_center = np.array([0, 0, 0])
        scene_min = np.array([-1, -1, -1])
        scene_max = np.array([1, 1, 1])
        max_extent = 2.0
    
    print(f"Scene center: {scene_center}, Max extent: {max_extent:.2f}")
    
    # Verify the transformation matrix and find its Euler angle representation
    from scipy.spatial.transform import Rotation
    transform_matrix = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    rot = Rotation.from_matrix(transform_matrix)
    euler_xyz = rot.as_euler('xyz', degrees=True)
    euler_XYZ = rot.as_euler('XYZ', degrees=True)
    print(f"\n=== Coordinate Frame Transformation ===")
    print(f"Transformation matrix (3x3):\n{transform_matrix}")
    print(f"As Euler angles (intrinsic xyz): {euler_xyz}")
    print(f"As Euler angles (extrinsic XYZ): {euler_XYZ}")
    
    # Create 4x4 homogeneous transformation matrix
    # This can transform both position and orientation together
    T_homogeneous = np.eye(4)
    T_homogeneous[:3, :3] = transform_matrix
    print(f"\nHomogeneous transformation (4x4):\n{T_homogeneous}")
    print(f"\nUsage for simulation:")
    print(f"  - Position: pos_new = T_homogeneous[:3, :3] @ pos_old")
    print(f"  - Orientation: R_new = T_homogeneous[:3, :3] @ R_old @ T_homogeneous[:3, :3].T")
    print(f"  - Full pose (4x4): pose_new = T_homogeneous @ pose_old @ T_homogeneous.T")
    print(f"=====================================\n")
    
    # Setup matplotlib figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits based on actual transformed vertex ranges
    # Add some margin (10% of range on each side)
    margin_factor = 0.1
    x_margin = max(scene_extent[0] * margin_factor, 0.05)
    y_margin = max(scene_extent[1] * margin_factor, 0.05)
    z_margin = max(scene_extent[2] * margin_factor, 0.05)
    
    ax.set_xlim(scene_min[0] - x_margin, scene_max[0] + x_margin)
    ax.set_ylim(scene_min[1] - y_margin, scene_max[1] + y_margin)
    ax.set_zlim(scene_min[2] - z_margin, scene_max[2] + z_margin)
    
    print(f"Axis limits:")
    print(f"  X: [{scene_min[0] - x_margin:.3f}, {scene_max[0] + x_margin:.3f}]")
    print(f"  Y: [{scene_min[1] - y_margin:.3f}, {scene_max[1] + y_margin:.3f}]")
    print(f"  Z: [{scene_min[2] - z_margin:.3f}, {scene_max[2] + z_margin:.3f}]")
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Hand and Object Vertices (Camera View)')
    
    # Set the view angle based on camera convention:
    # Standard camera: looks along -Z axis, with X right, Y down, Z forward
    # We want: X up, Y right, Z into screen
    # This means we need to flip Y (so Y is up becomes Y is down) and swap X<->Y
    # elev=0 (horizontal), azim=-90 (looking from +X towards -X), roll=0
    # Actually: camera at origin looking in +Z direction, X=right, Y=down in camera
    # To get X=up, Y=right: we look from -Z direction (azim=0), then rotate view
    ax.view_init(elev=0, azim=0, roll=0)
    
    # Optionally compute and display field of view
    fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    print(f"Field of view: {fov_x:.2f}° (horizontal), {fov_y:.2f}° (vertical)")
    
    # Store all frame data for animation
    frame_data = []
    all_transformed_vertices = []  # Collect all transformed vertices to check bounds
    
    for i, rgb_file in enumerate(tqdm(rgb_files, desc="Loading mesh data")):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        
        # # Load hand mesh files
        # hand_mesh_pattern = os.path.join(hand_params_dir, f"{image_name}_right_3d_*.obj")
        # hand_mesh_files = glob.glob(hand_mesh_pattern)
        
        # # Load object mesh
        # object_mesh_path = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
        if (optimized):
            hand_mesh_files = [os.path.join(object_dir, f"{image_name}_optimized_hand_mesh.obj")]
            object_mesh_pattern = os.path.join(object_dir, f"{image_name}_object_mesh_*.obj")
            object_mesh_files = glob.glob(object_mesh_pattern)
        
        else:
            # Load hand mesh files
            hand_mesh_pattern = os.path.join(hand_params_dir, f"{image_name}_right_3d_*.obj")
            hand_mesh_files = glob.glob(hand_mesh_pattern)
            # Load object mesh
            object_mesh_pattern = os.path.join(object_dir, f"{image_name}_object_mesh_*.obj")
            object_mesh_files = glob.glob(object_mesh_pattern)
        
        # Collect vertices for this frame
        hand_vertices = []
        object_vertices = []
        
        # Load hand mesh vertices
        for hand_mesh_file in hand_mesh_files:
            try:
                hand_mesh = trimesh.load(hand_mesh_file, force='mesh')
                if hand_mesh.vertices.shape[0] > 0:
                    # Sample every 10th vertex to avoid too many points
                    vertices = hand_mesh.vertices #[::10]
                    
                    # Transform vertices to match desired view: X=up, Y=right, Z=forward (into screen)
                    # Original camera coords: X=right, Y=down, Z=forward
                    # Target: X=up, Y=right, Z=forward
                    # Manual: X_new = -Z_old, Y_new = X_old, Z_new = -Y_old
                    from scipy.spatial.transform import Rotation
                    
                    # Use the exact transformation matrix: [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
                    transform_matrix = np.array([[0, 0, -1], 
                                                [1, 0, 0], 
                                                [0, -1, 0]])
                    rotation = Rotation.from_matrix(transform_matrix)
                    vertices_transformed = rotation.apply(vertices)
                    
                    hand_vertices.append(vertices_transformed)
                    all_transformed_vertices.append(vertices_transformed)
                    # # Apply rotation transformations: intrinsic rotation (Z then X in local frame)
                    # from scipy.spatial.transform import Rotation
                    
                    # # Create intrinsic rotation: first rotate around Z, then around X (in the rotated frame)
                    # # This is equivalent to composing rotations: R = Rz * Rx
                    # rot_z = Rotation.from_euler('z', 90, degrees=True)
                    # rot_x = Rotation.from_euler('x', 180, degrees=True)
                    
                    # # Intrinsic rotation: compose rotations and apply once
                    # combined_rotation = rot_z * rot_x  # This gives intrinsic rotation
                    # vertices_rotated = combined_rotation.apply(vertices)
                    
                    # hand_vertices.append(vertices_rotated)
            except Exception as e:
                print(f"Error loading hand mesh {hand_mesh_file}: {e}")
        
        # Load object mesh vertices
        for object_mesh_file in object_mesh_files:
            try:
                obj_mesh = trimesh.load(object_mesh_file, force='mesh')
                if obj_mesh.vertices.shape[0] > 0:
                    # Sample every 10th vertex to avoid too many points
                    vertices = obj_mesh.vertices[::10]
                    
                    # Transform vertices to match desired view: X=up, Y=right, Z=forward (into screen)
                    # Original camera coords: X=right, Y=down, Z=forward
                    # Target: X=up, Y=right, Z=forward
                    # Manual: X_new = -Z_old, Y_new = X_old, Z_new = -Y_old
                    from scipy.spatial.transform import Rotation
                    
                    # Use the exact transformation matrix: [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
                    transform_matrix = np.array([[0, 0, -1], 
                                                [1, 0, 0], 
                                                [0, -1, 0]])
                    rotation = Rotation.from_matrix(transform_matrix)
                    vertices_transformed = rotation.apply(vertices)
                    
                    object_vertices.append(vertices_transformed)
                    all_transformed_vertices.append(vertices_transformed)
                    # # Apply rotation transformations: intrinsic rotation (Z then X in local frame)
                    # from scipy.spatial.transform import Rotation
                    
                    # # Create intrinsic rotation: first rotate around Z, then around X (in the rotated frame)
                    # # This is equivalent to composing rotations: R = Rz * Rx
                    # rot_z = Rotation.from_euler('z', 90, degrees=True)
                    # rot_x = Rotation.from_euler('x', 180, degrees=True)
                    
                    # # Intrinsic rotation: compose rotations and apply once
                    # combined_rotation = rot_z * rot_x  # This gives intrinsic rotation
                    # vertices_rotated = combined_rotation.apply(vertices)
                    
                    # object_vertices.append(vertices_rotated)
            except Exception as e:
                print(f"Error loading object mesh {object_mesh_path}: {e}")
        
        # Store frame data
        frame_data.append({
            'hand_vertices': hand_vertices,
            'object_vertices': object_vertices,
            'frame_name': image_name
        })
    
    # Verify bounds across all frames
    if all_transformed_vertices:
        all_verts = np.vstack(all_transformed_vertices)
        actual_min = np.min(all_verts, axis=0)
        actual_max = np.max(all_verts, axis=0)
        print(f"\nActual transformed vertices range across all frames:")
        print(f"  X: [{actual_min[0]:.3f}, {actual_max[0]:.3f}]")
        print(f"  Y: [{actual_min[1]:.3f}, {actual_max[1]:.3f}]")
        print(f"  Z: [{actual_min[2]:.3f}, {actual_max[2]:.3f}]")
        
        # Update bounds if they differ significantly from first frame
        if not np.allclose(actual_min, scene_min, atol=0.01) or not np.allclose(actual_max, scene_max, atol=0.01):
            print("\nUpdating bounds to match all frames...")
            scene_min = actual_min
            scene_max = actual_max
            scene_extent = scene_max - scene_min
            
            # Recalculate margins
            x_margin = max(scene_extent[0] * margin_factor, 0.05)
            y_margin = max(scene_extent[1] * margin_factor, 0.05)
            z_margin = max(scene_extent[2] * margin_factor, 0.05)
            
            # Update axis limits
            ax.set_xlim(scene_min[0] - x_margin, scene_max[0] + x_margin)
            ax.set_ylim(scene_min[1] - y_margin, scene_max[1] + y_margin)
            ax.set_zlim(scene_min[2] - z_margin, scene_max[2] + z_margin)
            
            print(f"Updated axis limits:")
            print(f"  X: [{scene_min[0] - x_margin:.3f}, {scene_max[0] + x_margin:.3f}]")
            print(f"  Y: [{scene_min[1] - y_margin:.3f}, {scene_max[1] + y_margin:.3f}]")
            print(f"  Z: [{scene_min[2] - z_margin:.3f}, {scene_max[2] + z_margin:.3f}]")
    
    # Create animation
    def animate(frame_idx):
        ax.clear()
        
        # Set axis properties again with fixed limits (same as initial setup)
        ax.set_xlim(scene_min[0] - x_margin, scene_max[0] + x_margin)
        ax.set_ylim(scene_min[1] - y_margin, scene_max[1] + y_margin)
        ax.set_zlim(scene_min[2] - z_margin, scene_max[2] + z_margin)
        
        ax.set_xlabel('X (Up)')
        ax.set_ylabel('Y (Right)')
        ax.set_zlabel('Z (Forward)')
        ax.set_title(f'3D Hand and Object Vertices - Frame {frame_idx} (Camera View)')
        
        # Set the view angle to match camera convention
        ax.view_init(elev=0, azim=0, roll=0)
        
        if frame_idx < len(frame_data):
            frame = frame_data[frame_idx]
            
            # Plot hand vertices in blue
            for hand_verts in frame['hand_vertices']:
                if len(hand_verts) > 0:
                    ax.scatter(hand_verts[:, 0], hand_verts[:, 1], hand_verts[:, 2], 
                              c='blue', s=20, alpha=0.7, label='Hand' if len(frame['hand_vertices']) == 1 else "")
            
            # Plot object vertices in red
            for obj_verts in frame['object_vertices']:
                if len(obj_verts) > 0:
                    ax.scatter(obj_verts[:, 0], obj_verts[:, 1], obj_verts[:, 2], 
                              c='red', s=20, alpha=0.7, label='Object' if len(frame['object_vertices']) == 1 else "")
        
        # Add legend only once
        if frame_idx == 0:
            ax.legend()
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(frame_data), 
                                  interval=100, blit=False, repeat=True)
    
    # Save animation as video
    print("Saving animation as video...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(video_path, writer=writer)
    
    plt.close(fig)
    print(f"3D vertices video saved: {video_path}")

def meshes_3d_video(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir, optimized = False, robot_qpos_dir = None):
    """
    Render 3D meshes using camera intrinsics K and save as video
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) #[:10]
    robot_qpos_index_path = os.path.join(robot_qpos_dir, "robot_qpos_index.npy")
    robot_qpos_index_all = np.load(robot_qpos_index_path)
    rgb_files = [rgb_files[i] for i in robot_qpos_index_all]
    # Load camera intrinsics
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    # Extract camera parameters from K matrix
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point
    
    # Setup video writer
    if (optimized):
        video_path = os.path.join(output_dir, f"human_optimized.mp4")
    else:
        video_path = os.path.join(output_dir, f"human.mp4")
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Get video dimensions from first frame
    first_frame = cv2.imread(rgb_files[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Setup renderer for 3D mesh rendering
    renderer = pyrender.OffscreenRenderer(width, height)
    
    print(f"Creating 3D video: {video_path}")
    print(f"Camera intrinsics - fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")
    
    # Calculate fixed camera pose using first frame
    first_rgb_file = rgb_files[0]
    first_image_name = os.path.basename(first_rgb_file).replace(".png", "")
    if (optimized):
        first_hand_files = [os.path.join(object_dir, f"{first_image_name}_optimized_hand_mesh.obj")]
        first_object_path = os.path.join(object_dir, f"{first_image_name}_optimized_object_mesh.obj")
    else:
        # Load meshes from first frame to calculate fixed camera position
        first_hand_pattern = os.path.join(hand_params_dir, f"{first_image_name}_right_3d_*.obj")
        first_hand_files = glob.glob(first_hand_pattern)
        first_object_path = os.path.join(object_dir, f"{first_image_name}_object_mesh.obj")
    
    # # Calculate scene center from first frame
    # first_frame_vertices = []
    # for hand_mesh_file in first_hand_files:
    #     try:
    #         hand_mesh = trimesh.load(hand_mesh_file, force='mesh')
    #         if hand_mesh.vertices.shape[0] > 0:
    #             first_frame_vertices.append(hand_mesh.vertices)
    #     except Exception as e:
    #         print(f"Error loading first frame hand mesh {hand_mesh_file}: {e}")
    
    # if os.path.exists(first_object_path):
    #     try:
    #         obj_mesh = trimesh.load(first_object_path, force='mesh')
    #         if obj_mesh.vertices.shape[0] > 0:
    #             first_frame_vertices.append(obj_mesh.vertices)
    #     except Exception as e:
    #         print(f"Error loading first frame object mesh {first_object_path}: {e}")
    
    # # Calculate fixed scene center and camera pose
    # if first_frame_vertices:
    #     all_first_vertices = np.vstack(first_frame_vertices)
    #     fixed_scene_center = np.mean(all_first_vertices, axis=0)
    #     scene_extent = np.max(all_first_vertices, axis=0) - np.min(all_first_vertices, axis=0)
    #     max_extent = np.max(scene_extent)
    # else:
    #     fixed_scene_center = np.array([0, 0, 0])
    #     max_extent = 2.0
    
    # camera_distance = 1.5 * max_extent 
    # fixed_camera_pose = np.eye(4)
    # fixed_camera_pose[2, 3] = fixed_scene_center[2] + camera_distance  # Move camera back
    # fixed_camera_pose[1, 3] = fixed_scene_center[1] #- camera_distance * 1 #0.3  # Move camera down slightly
    # fixed_camera_pose[0, 3] = fixed_scene_center[0]  # Center camera horizontally

    # Create fixed camera pose
    # Use camera at origin, looking along -Z (OpenCV to pyrender convention)
    fixed_camera_pose = np.eye(4)
    fixed_camera_pose[2, 2] = -1.0
    
    print(f"Fixed camera pose calculated from first frame")
    # print(f"Scene center: {fixed_scene_center}, Camera distance: {camera_distance:.2f}")
    
    for i, rgb_file in enumerate(tqdm(rgb_files, desc="Rendering 3D frames")):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        
        if (optimized):
            hand_mesh_files = [os.path.join(object_dir, f"{image_name}_optimized_hand_mesh.obj")]
            object_mesh_path = os.path.join(object_dir, f"{image_name}_optimized_object_mesh.obj")
        else:
            # Load hand mesh files
            hand_mesh_pattern = os.path.join(hand_params_dir, f"{image_name}_right_3d_*.obj")
            hand_mesh_files = glob.glob(hand_mesh_pattern)
            # Load object mesh
            object_mesh_pattern = os.path.join(object_dir, f"{image_name}_object_mesh_*.obj")
            object_mesh_files = glob.glob(object_mesh_pattern)
        
        # Load hand meshes
        hand_meshes = []
        for hand_mesh_file in hand_mesh_files:
            try:
                hand_mesh = trimesh.load(hand_mesh_file, force='mesh')
                if hand_mesh.vertices.shape[0] > 0:
                    hand_meshes.append(hand_mesh)
            except Exception as e:
                print(f"Error loading hand mesh {hand_mesh_file}: {e}")
        
        # Load object mesh
        obj_meshes = []
        for object_mesh_file in object_mesh_files:
            try:
                obj_mesh = trimesh.load(object_mesh_file, force='mesh')
                if obj_mesh.vertices.shape[0] > 0:
                    obj_meshes.append(obj_mesh)
            except Exception as e:
                print(f"Error loading object mesh {object_mesh_file}: {e}")
        
        # Create scene
        scene = pyrender.Scene()
        
        # Create camera using the intrinsics K with fixed pose
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=fixed_camera_pose)
        
        # Add lighting (also fixed relative to scene)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.eye(4)
        # light_pose[2, 3] = fixed_scene_center[2] + 5.0
        # light_pose[1, 3] = fixed_scene_center[1] + 3.0
        scene.add(light, pose=light_pose)
        
        # Add point light for better visibility
        point_light = pyrender.PointLight(color=[0.3, 0.3, 0.3], intensity=1.0)
        point_light_pose = np.eye(4)
        # point_light_pose[2, 3] = fixed_scene_center[2] + 3.0
        scene.add(point_light, pose=point_light_pose)
        
        # Add hand meshes to scene
        for hand_mesh in hand_meshes:
            try:
                hand_mesh_pyrender = pyrender.Mesh.from_trimesh(hand_mesh)
                hand_color_normalized = np.array([102, 192, 255]) / 255.0
                hand_material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.5,
                    baseColorFactor=(hand_color_normalized[0], hand_color_normalized[1], hand_color_normalized[2], 0.8)
                )
                hand_mesh_pyrender.primitives[0].material = hand_material
                scene.add(hand_mesh_pyrender)
            except Exception as e:
                print(f"Error adding hand mesh to scene: {e}")
        
        # Add object mesh to scene
        for obj_mesh in obj_meshes:
            try:
                obj_mesh_pyrender = pyrender.Mesh.from_trimesh(obj_mesh)
                obj_color_normalized = np.array([239, 132, 167]) / 255.0
                obj_material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.5,
                    baseColorFactor=(obj_color_normalized[0], obj_color_normalized[1], obj_color_normalized[2], 1.0)
                )
                obj_mesh_pyrender.primitives[0].material = obj_material
                scene.add(obj_mesh_pyrender)
            except Exception as e:
                print(f"Error adding object mesh to scene: {e}")
        
        # Render scene
        try:
            color, depth = renderer.render(scene)
            
            # Convert from RGB to BGR for OpenCV
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # flip the color_bgr vertically
            color_bgr = cv2.flip(color_bgr, 0)
            # flip the color_bgr horizontally
            color_bgr = cv2.flip(color_bgr, 1)

            # Write frame to video
            video_writer.write(color_bgr)
            

            # color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # color_bgr = cv2.flip(color_bgr, 0)   # keep this line
            # video_writer.write(color_bgr)
            
        except Exception as e:
            print(f"Error rendering frame {image_name}: {e}")
            # Create a black frame if rendering fails
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            video_writer.write(black_frame)
    
    # Clean up
    # # print the total frame number of the video
    print(f"Total frame number of the video: {video_writer.get(cv2.CAP_PROP_FRAME_COUNT)}")
    
    video_writer.release()
    renderer.delete()
    print(f"3D video saved: {video_path}")





def robot_meshes_3d_video(folder_name, rgb_dir, hand_params_dir, robot_name, robot_qpos_dir, object_dir, output_dir, optimized = False):
    """
    Render 3D meshes using camera intrinsics K and save as video
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) #[:10]
    robot_qpos_index_path = os.path.join(robot_qpos_dir, "robot_qpos_index.npy")
    print(f"length of robot_qpos_index_all: {len(np.load(robot_qpos_index_path))}")
    robot_qpos_index_all = np.load(robot_qpos_index_path)
    rgb_files = [rgb_files[i] for i in robot_qpos_index_all]
    # Load camera intrinsics
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    # Extract camera parameters from K matrix
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point
    
    # Setup video writer
    if (optimized):
        video_path = os.path.join(output_dir, f"robot_optimized.mp4")
    else:
        video_path = os.path.join(output_dir, f"robot.mp4")
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Get video dimensions from first frame
    first_frame = cv2.imread(rgb_files[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Setup renderer for 3D mesh rendering
    renderer = pyrender.OffscreenRenderer(width, height)
    
    print(f"Creating 3D video: {video_path}")
    print(f"Camera intrinsics - fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")

    if (optimized):
        robot_qpos = np.load(os.path.join(robot_qpos_dir, f"robot_qpos_optimized.npy"))
    else:
        robot_qpos = np.load(os.path.join(robot_qpos_dir, f"robot_qpos.npy"))
    #pdb.set_trace()
    robot_hand_model = create_hand_model(robot_name, torch.device('cpu'))
    # Load hand meshes
    robot_hand_meshes = []
    for qpos in robot_qpos:
        try:
            qpos = torch.from_numpy(qpos).float()
            robot_hand_mesh = robot_hand_model.get_trimesh_q(qpos)['visual']
            if robot_hand_mesh.vertices.shape[0] > 0:
                robot_hand_meshes.append(robot_hand_mesh)
        except Exception as e:
            print(f"Error loading robot hand mesh {qpos}: {e}")

    # Create fixed camera pose
    # Use camera at origin, looking along -Z (OpenCV to pyrender convention)
    fixed_camera_pose = np.eye(4)
    fixed_camera_pose[2, 2] = -1.0

    print(f"Fixed camera pose calculated from first frame")
    # print(f"Scene center: {fixed_scene_center}, Camera distance: {camera_distance:.2f}")
    
    # Flag to track if viewer should be shown (disable after failures)
    # Set to False to disable viewer entirely
    show_viewer = True
    viewer_failure_count = 0
    max_viewer_failures = 1  # Stop showing viewer after 1 failure to prevent multiple windows
    
    for i, rgb_file in enumerate(tqdm(rgb_files, desc="Rendering 3D frames")):
        skip=False
        image_name = os.path.basename(rgb_file).replace(".png", "")
        #pdb.set_trace()
        # check if the image_name is in the human_hand folder
        if (optimized):
            hand_params_path = os.path.join(hand_params_dir, f"{image_name}_optimized_hand_params.npy")
        else:
            hand_params_path = os.path.join(hand_params_dir, f"{image_name}_mano_data.npy")


        # object_mesh_pattern = os.path.join(object_dir, f"{image_name}_object_mesh_*.obj")
        # object_mesh_files = glob.glob(object_mesh_pattern)

        # # Load object mesh
        # obj_meshes = []
        # for object_mesh_file in object_mesh_files:
        #     try:
        #         obj_mesh = trimesh.load(object_mesh_file, force='mesh')
        #         if obj_mesh.vertices.shape[0] > 0:
        #             obj_meshes.append(obj_mesh)
        #     except Exception as e:
        #         print(f"Error loading object mesh {object_mesh_file}: {e}")
        
        # Create scene
        scene = pyrender.Scene()
        
        # Create camera using the intrinsics K with fixed pose
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=fixed_camera_pose)
        
        # Add lighting (also fixed relative to scene)
        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=np.eye(4))
        point_light = pyrender.PointLight(color=[0.3, 0.3, 0.3], intensity=1.0)
        scene.add(point_light, pose=np.eye(4))
        
        # Add robot hand mesh (brighter light blue)
        # Hand color: (102, 192, 255) with opacity 0.8
        hand_color_normalized = np.array([102, 192, 255]) / 255.0
        robot_hand_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.5,
            baseColorFactor=(hand_color_normalized[0], hand_color_normalized[1], hand_color_normalized[2], 0.8)
        )
        robot_hand_mesh_pyrender = pyrender.Mesh.from_trimesh(robot_hand_meshes[i], material=robot_hand_material)
        scene.add(robot_hand_mesh_pyrender)
        
        # Add object meshes (pink color)
        # Object color: (239, 132, 167) with opacity 1.0
        obj_color_normalized = np.array([239, 132, 167]) / 255.0
        obj_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.5,
            baseColorFactor=(obj_color_normalized[0], obj_color_normalized[1], obj_color_normalized[2], 1.0)
        )
        # # Add all grasp object meshes
        # for obj_mesh in obj_meshes:
        #     obj_mesh_pyrender = pyrender.Mesh.from_trimesh(obj_mesh, material=obj_material)
        #     scene.add(obj_mesh_pyrender)
        
        # # Add hand meshes to scene
        # try:
        #     robot_hand_mesh_pyrender = pyrender.Mesh.from_trimesh(robot_hand_meshes[i])
        #     scene.add(robot_hand_mesh_pyrender)
            
        #     # Sample point cloud from robot hand mesh
        #     # robot_hand_pc = robot_hand_meshes[i-skip_count].sample(512)  # Sample 512 points
            
        #     # # Create spheres for each point in the robot hand point cloud
        #     # for point in robot_hand_pc:
        #     #     sphere = trimesh.creation.icosphere(subdivisions=0, radius=0.002)  # Small radius
        #     #     sphere.apply_translation(point)
        #     #     robot_hand_sphere_pyrender = pyrender.Mesh.from_trimesh(sphere, smooth=False)
        #     #     # Add red material
        #     #     robot_hand_sphere_pyrender.primitives[0].material = pyrender.MetallicRoughnessMaterial(
        #     #         baseColorFactor=[1.0, 0.0, 0.0, 1.0],  # Red color
        #     #         metallicFactor=0.0,
        #     #         roughnessFactor=0.5
        #     #     )
        #     #     scene.add(robot_hand_sphere_pyrender)
            
        # except Exception as e:
        #     print(f"Error adding robot hand mesh to scene: {e}")
        
        # # Add object mesh to scene
        # for obj_mesh in obj_meshes:
        #     try:
        #         obj_mesh_pyrender = pyrender.Mesh.from_trimesh(obj_mesh)
        #         scene.add(obj_mesh_pyrender)
                
        #         # # Sample point cloud from object mesh
        #         # object_pc = obj_mesh.sample(512)  # Sample 512 points
                
        #         # # Create spheres for each point in the object point cloud
        #         # for point in object_pc:
        #         #     sphere = trimesh.creation.icosphere(subdivisions=0, radius=0.002)  # Small radius
        #         #     sphere.apply_translation(point)
        #         #     object_sphere_pyrender = pyrender.Mesh.from_trimesh(sphere, smooth=False)
        #         #     # Add blue material
        #         #     object_sphere_pyrender.primitives[0].material = pyrender.MetallicRoughnessMaterial(
        #         #         baseColorFactor=[0.0, 0.0, 1.0, 1.0],  # Blue color
        #         #         metallicFactor=0.0,
        #         #         roughnessFactor=0.5
        #         #     )
        #         #     scene.add(object_sphere_pyrender)
                
        #     except Exception as e:
        #         print(f"Error adding object mesh to scene: {e}")
            
        # Render scene
        try:
            color, depth = renderer.render(scene)
            
            # Convert from RGB to BGR for OpenCV
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # flip the color_bgr vertically
            color_bgr = cv2.flip(color_bgr, 0)
            # flip the color_bgr horizontally
            color_bgr = cv2.flip(color_bgr, 1)
            # save the color_bgr to a png file
            # print(f"image_name: {image_name}, robot qpos: {robot_qpos[i, :6]}")
            # cv2.imwrite(os.path.join("<repo>/visualization/wild_5_bulb", f"{image_name}.png"), color_bgr)
            # Write frame to video
            video_writer.write(color_bgr)
            
        except Exception as e:
            print(f"Error rendering frame {image_name}: {e}")
            # Create a black frame if rendering fails
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            video_writer.write(black_frame)
    
    # Clean up
    video_writer.release()
    renderer.delete()
    print(f"3D video saved: {video_path}")


def test_viewer(folder_name, rgb_dir, hand_params_dir, robot_name, robot_qpos_dir, object_dir, output_dir, optimized = False, obj = None):
    """
    Render 3D meshes using camera intrinsics K and save as video
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) #[:10]
    robot_qpos_index_path = os.path.join(robot_qpos_dir, "robot_qpos_index.npy")
    robot_qpos_index_all = np.load(robot_qpos_index_path)
    # rgb_files = [rgb_files[i] for i in robot_qpos_index_all]
    # Load camera intrinsics
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    # Extract camera parameters from K matrix
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point
    
    # Setup video writer
    if (optimized):
        video_path = os.path.join(output_dir, f"test.mp4")
    else:
        video_path = os.path.join(output_dir, f"test.mp4")
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Get video dimensions from first frame
    first_frame = cv2.imread(rgb_files[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Setup renderer for 3D mesh rendering
    renderer = pyrender.OffscreenRenderer(width, height)
    
    print(f"Creating 3D video: {video_path}")
    print(f"Camera intrinsics - fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")

    if (optimized):
        robot_qpos = np.load(os.path.join(robot_qpos_dir, f"robot_qpos_optimized.npy"))
    else:
        robot_qpos = np.load(os.path.join(robot_qpos_dir, f"robot_qpos.npy"))
    #pdb.set_trace()
    robot_hand_model = create_hand_model(robot_name, torch.device('cpu'))
    # Load hand meshes
    robot_hand_meshes = []
    for qpos in robot_qpos:
        try:
            qpos = torch.from_numpy(qpos).float()
            robot_hand_mesh = robot_hand_model.get_trimesh_q(qpos)['visual']
            if robot_hand_mesh.vertices.shape[0] > 0:
                robot_hand_meshes.append(robot_hand_mesh)
        except Exception as e:
            print(f"Error loading robot hand mesh {qpos}: {e}")

    # Create fixed camera pose
    # Use camera at origin, looking along -Z (OpenCV to pyrender convention)
    fixed_camera_pose = np.eye(4)
    fixed_camera_pose[2, 2] = -1.0

    print(f"Fixed camera pose calculated from first frame")
    # print(f"Scene center: {fixed_scene_center}, Camera distance: {camera_distance:.2f}")

    for i, rgb_file in enumerate(tqdm(rgb_files, desc="Rendering 3D frames")):
        skip=False
        image_name = os.path.basename(rgb_file).replace(".png", "")
        #pdb.set_trace()
        # check if the image_name is in the human_hand folder
        if (optimized):
            hand_params_path = os.path.join(hand_params_dir, f"{image_name}_optimized_hand_params.npy")
        else:
            hand_params_path = os.path.join(hand_params_dir, f"{image_name}_mano_data.npy")


        object_mesh_pattern = os.path.join(object_dir, f"{image_name}_object_mesh_*.obj")
        object_mesh_files = glob.glob(object_mesh_pattern)

        # Load object mesh
        obj_meshes = []
        for object_mesh_file in object_mesh_files:
            try:
                obj_mesh = trimesh.load(object_mesh_file, force='mesh')
                if obj_mesh.vertices.shape[0] > 0:
                    obj_meshes.append(obj_mesh)
            except Exception as e:
                print(f"Error loading object mesh {object_mesh_file}: {e}")
        
        # Create scene
        scene = trimesh.Scene()
        
        # Create camera using the intrinsics K with fixed pose
        
        # Add hand meshes to scene
        try:
            scene.add_geometry(robot_hand_meshes[i])
            

            
        except Exception as e:
            print(f"Error adding robot hand mesh to scene: {e}")
        
        # Add object mesh to scene
        for obj_mesh in obj_meshes:
            try:
                scene.add_geometry(obj_mesh)
            
                
            except Exception as e:
                print(f"Error adding object mesh to scene: {e}")
            
        if i % 5 ==0: 
            os.makedirs(f"./visualization/viewer_scenes/{obj}", exist_ok=True)
            # save the scene to a png file
            # png_data = scene.save_image(resolution=(1080,1080))
            # file_name = f"scene_render_{i}.png"
            # with open(os.path.join(f"./visualization/viewer_scenes/{obj}", file_name), 'wb') as f:
            #     f.write(png_data)
            scene.export(os.path.join(f"./visualization/viewer_scenes/{obj}", f"{i}_scene.obj"))
            # pdb.set_trace()

def compile_rgb2video(folder_name, rgb_dir, output_dir, robot_qpos_dir):
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    robot_qpos_index_path = os.path.join(robot_qpos_dir, "robot_qpos_index.npy")
    robot_qpos_index_all = np.load(robot_qpos_index_path)
    rgb_files = [rgb_files[i] for i in robot_qpos_index_all]
    video_path = os.path.join(output_dir, f"rgb.mp4")
    first_frame = cv2.imread(rgb_files[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for rgb_file in tqdm(rgb_files, desc="Compiling RGB to video"):
        rgb = cv2.imread(rgb_file)
        video_writer.write(rgb)
    video_writer.release()


def extract_frames_from_video(video_path, robot_qpos_index_path, output_dir, output_video_name=None):
    """
    Extract frames from a video using indices from robot_qpos_index_all and save as a new video.
    
    Args:
        video_path: Path to the input video file
        robot_qpos_index_path: Path to robot_qpos_index.npy file containing frame indices
        output_dir: Directory to save the output video
        output_video_name: Name of the output video file (if None, auto-generate from input video name)
    """
    # Load frame indices
    robot_qpos_index_all = np.load(robot_qpos_index_path)
    print(f"Extracting {len(robot_qpos_index_all)} frames from video: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {total_frames} frames, {fps} fps, {width}x{height}")
    
    # Generate output video path
    if output_video_name is None:
        input_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_video_name = f"{input_basename}_extracted.mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Extract frames at specified indices
    extracted_count = 0
    for frame_idx in tqdm(robot_qpos_index_all, desc="Extracting frames"):
        if frame_idx >= total_frames:
            print(f"Warning: Frame index {frame_idx} exceeds video length ({total_frames}), skipping")
            continue
        
        # Set video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Write frame to output video
            video_writer.write(frame)
            extracted_count += 1
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    # # print the total frame number of the video
    print(f"Total frame number of the extracted video: {extracted_count}")
    cap.release()
    video_writer.release()
    print(f"Successfully extracted {extracted_count} frames and saved to: {output_video_path}")

def main():
    # parser = argparse.ArgumentParser(description='Visualize object and hand meshes per frame')
    # parser.add_argument('--object_dir', default="<data_root>/clean_surface", type=str, help='Directory containing *_object_mesh.obj and *_hand_mesh.obj files')
    # all_objects = ["declutter_desk_cup"]  #"color_with_pen", "flip_coin", 
    all_objects = ['real_14_pourtea'] #['wild_12_hang', 'wild_14_bulb', 'sora_1_jenga', 'real_16_closedrawer']
    #['real_14_pourtea'] #, "real_19_placebottle", "real_16_closedrawer"] #'real_14_pourtea'
    #['train_bottle_1', 'train_hanger_2', 'train_hat_3', 'train_spraybottle_4', 'train_mug_5', 'train_bowl_6', 'train_pot_8', 'train_pan_8', 'train_ladle_9', 'train_sprayer_10', 'train_toothbrush_11', 'train_cup_12', 'train_scissors_13', 'train_case_14', 'train_apple_17', 'train_umbrella_18', 'train_handlebag_19', 'train_sunglass_20', 'train_powerdrill_21', 'train_wineglass_25', 'train_hat_3_4', 'train_hat_3_5', 'train_pan_8_2', 'train_pan_8_1']

    # ['train_hat_3_4', 'train_hat_3_5', 'train_pan_8_2', 'train_pan_8_1']
    #
    #['sora_0_pour', 'real_14_pourtea', 'real_19_placebottle', 'wild_0_pourtea'] #'youtube_11_nugget' 'wild_10_jenga', 
    #['train_handlebag_19_1'] #, 'train_handlebag_19_2']
    #['youtube_11_nugget']
    #['youtube_7_cook', 'youtube_8_fries'] #'youtube_3_clean', 'youtube_4_cook', 
    #['real_19_placebottle', 'real_16_closedrawer', 'real_14_pourtea']
    #['train_bottle_1', 'train_hanger_2', 'train_hat_3', 'train_spraybottle_4', 'train_mug_5', 'train_bowl_6', 'train_pot_8', 'train_pan_8', 'train_ladle_9', 'train_sprayer_10', 'train_toothbrush_11', 'train_cup_12', 'train_scissors_13', 'train_case_14', 'train_umbrella_18', 'train_handlebag_19', 'train_sunglass_20'] #["fry_egg_add_oil", "fry_egg_take_pan"]
    #["color_with_pen", "flip_coin", "clean_surface", "fry_egg_add_oil", "fry_egg_take_pan"]
    # Per-object working dirs; process_videos.sh exports DATA_ROOT.
    FOLDER_DIR = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    use_optimize = False
    for obj in all_objects:
        print("-" * 50)
        print(f"start rendering for {obj}")
        folder_name = os.path.join(FOLDER_DIR, obj)
        rgb_dir = os.path.join(folder_name, "rgb")
        if use_optimize:
            hand_params_dir = os.path.join(folder_name, "optimized_hand_object")
            object_dir = os.path.join(folder_name, "obj_mesh")
        else:
            hand_params_dir = os.path.join(folder_name, "human_hand")
            object_dir = os.path.join(folder_name, "obj_mesh")
        robot_name = "inspire" #"inspire" #"leaphand"
        robot_qpos_dir = os.path.join(folder_name, "robot_qpos", robot_name)
        output_dir = os.path.join(folder_name, "videos")
        os.makedirs(output_dir, exist_ok=True)

        # # rgb video
        
        # # print("Waiting for 10 minutes...")
        # # time.sleep(60*10)
        # # delete the videos in the output_dir
        # for file in os.listdir(output_dir):
        #     if file.endswith(".mp4"):
        #         os.remove(os.path.join(output_dir, file))

        # extract_frames_from_video("<repo>/FoundationPose/debug_multi_scale/train_pan_8_1/grasp/temp_scale_0.75/foundationpose_scale_0.75.mp4", os.path.join(robot_qpos_dir, "robot_qpos_index.npy"), "<repo>")

        # "<repo>/FoundationPose/debug_multi_scale/youtube_1_pour/grasp/temp_scale_0.90"
        #extract_frames_from_video("<repo>/FoundationPose/debug_multi_scale/youtube_1_pour/grasp/temp_scale_0.90/foundationpose_scale_0.90.mp4", os.path.join(robot_qpos_dir, "robot_qpos_index.npy"), "<repo>")
        compile_rgb2video(folder_name, rgb_dir, output_dir, robot_qpos_dir)
        # project_meshes_to_2d_video(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir, robot_qpos_dir)
        # # # # human meshes 3d video
        # meshes_3d_video(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir, optimized = use_optimize, robot_qpos_dir = robot_qpos_dir)
        # # # # # robot meshes 3d video

        # robot_meshes_3d_video(folder_name, rgb_dir, hand_params_dir, robot_name, robot_qpos_dir, object_dir, output_dir, optimized = use_optimize)
        # # # Create both 3D mesh video and 2D projected video
        
        # render_2d_vertices_video_manual(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir)
        # render_3d_vertices_video(folder_name, rgb_dir, hand_params_dir, object_dir, output_dir, optimized = use_optimize)
        
        # test_viewer(folder_name, rgb_dir, hand_params_dir, robot_name, robot_qpos_dir, object_dir, output_dir, optimized = use_optimize, obj = obj)
        print(f"Completed rendering for {obj}")

if __name__ == "__main__":
    main() 
