#!/usr/bin/env python3
"""
Multi-scale FoundationPose demo that tries different mesh scales and picks the best one.
"""

import os
import torch
from estimater import *
from datareader import *
import argparse
import pdb 
import cv2
from tqdm import tqdm
import numpy as np
import trimesh
import shutil
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings,
    MeshRenderer, MeshRasterizer,
    SoftSilhouetteShader, BlendParams,
)
from PIL import Image
import pdb 

def resize_mesh_for_testing(mesh, scale_factor, output_dir, original_mesh_dir=None):
    """
    Create a temporary resized mesh for testing, preserving textures if they exist.
    
    Args:
        mesh: Original trimesh object
        scale_factor: Scale factor to apply
        output_dir: Directory to save the resized mesh
        original_mesh_dir: Directory containing original mesh and texture files
    
    Returns:
        tuple: (resized_mesh, mesh_path)
    """
    # Create a copy of the mesh
    resized_mesh = mesh.copy()
    resized_mesh.vertices *= scale_factor
    
    # Save to temporary location
    os.makedirs(output_dir, exist_ok=True)
    mesh_name = f"temp_mesh_scale_{scale_factor:.2f}"
    mesh_path = os.path.join(output_dir, f"{mesh_name}.obj")
    
    # Check if mesh has texture and preserve it
    has_texture = isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
    
    if has_texture:
        # If mesh has texture, export with texture
        # The export should preserve the texture reference
        resized_mesh.export(mesh_path, include_texture=True)
        
        # Copy texture files if original_mesh_dir is provided
        if original_mesh_dir is not None:
            # Find and copy texture-related files (.mtl, .png, .jpg, etc.)
            base_name = os.path.splitext(os.path.basename(mesh_path))[0]
            for file in os.listdir(original_mesh_dir):
                # Look for texture files that might be referenced
                if file.endswith(('.mtl', '.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(original_mesh_dir, file)
                    # Copy with the same name to maintain references
                    dst_path = os.path.join(output_dir, file)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied texture file: {file} to {output_dir}")
    else:
        # No texture, just export the mesh
        resized_mesh.export(mesh_path)
    # check if the resized_mesh has texture
    if isinstance(resized_mesh.visual, trimesh.visual.texture.TextureVisuals):
        # check if the texture is the same as the original mesh
        if resized_mesh.visual.material.image == mesh.visual.material.image:
            print(f"Resized mesh texture is the same as the original mesh")
        else:
            print(f"Resized mesh texture is not the same as the original mesh")
        # pdb.set_trace()
        print(f"Resized mesh has texture")
    else:
        print(f"Resized mesh does not have texture")
    # pdb.set_trace()
    return resized_mesh, mesh_path


def test_single_scale(original_mesh, scale_factor, reader, debug_dir, est_refine_iter=5, track_refine_iter=2, original_mesh_dir=None):
    """
    Test pose estimation with a single mesh scale and save rendering video.
    
    Args:
        mesh: Original mesh object
        scale_factor: Scale factor to test
        reader: Data reader object
        debug_dir: Debug directory
        est_refine_iter: Registration refinement iterations
        track_refine_iter: Tracking refinement iterations
    
    Returns:
        dict: Results including scores and poses
    """
    print(f"\n=== Testing scale factor: {scale_factor} ===")
    
    # Create temporary directory for this scale
    temp_dir = os.path.join(debug_dir, f"temp_scale_{scale_factor:.2f}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Resize mesh

    resized_mesh, mesh_path = resize_mesh_for_testing(original_mesh, scale_factor, temp_dir, original_mesh_dir=original_mesh_dir)
    
    # Initialize FoundationPose with resized mesh
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    est = FoundationPose(
        model_pts=resized_mesh.vertices, 
        model_normals=resized_mesh.vertex_normals, 
        mesh=resized_mesh, 
        scorer=scorer, 
        refiner=refiner, 
        debug_dir=temp_dir, 
        debug=0,  # Reduce debug output for speed
        glctx=glctx
    )
    
    # Initialize video writer for this scale
    fps = 30
    first_color = reader.get_color(0)
    frame_size = (first_color.shape[1], first_color.shape[0])
    video_path = os.path.join(temp_dir, f'foundationpose_scale_{scale_factor:.2f}.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    
    if not video_writer.isOpened():
        print(f"Could not open video writer for {video_path}")
        return {
            'scale_factor': scale_factor,
            'avg_score': -1.0,
            'scores': [],
            'poses': [],
            'mesh_path': None,
            'resized_mesh': None,
            'error': 'Could not open video writer'
        }
    
    # Get mesh bounds for visualization
    to_origin, extents = trimesh.bounds.oriented_bounds(resized_mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    # Test on first few frames
    num_test_frames = len(reader.color_files) #min(len(reader.color_files), 20) #len(reader.color_files) #
    scores = []
    poses = []
    camera_object_meshes = []
    rendering_error = 0.0
    results_file = os.path.join(temp_dir, f"mask_error.txt")
    # Define device for PyTorch operations
    device = torch.device("cuda:0")
    for i in range(num_test_frames):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        
        if i == 0:
            # Registration phase
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
            
            # Get registration score
            if hasattr(est, 'scores') and len(est.scores) > 0:
                reg_score = est.scores[0].item()  # Best score from registration
            else:
                reg_score = 0.0
            scores.append(reg_score)
            poses.append(pose)
            
        else:
            # Tracking phase
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
            
            # For tracking, we don't have explicit scores, so we'll use a heuristic
            # based on pose consistency
            if len(poses) > 0:
                # Calculate pose change (simple heuristic)
                pose_change = np.linalg.norm(pose[:3, 3] - poses[-1][:3, 3])
                track_score = max(0.0, float(1.0 - pose_change))  # Higher score for smaller changes
            else:
                track_score = 0.0
            scores.append(track_score)
            poses.append(pose)
        
        # Create visualization frame
        center_pose = pose @ np.linalg.inv(to_origin)
        # First draw the coordinate axes on a copy of the image
        coord_vis = draw_posed_3d_box(reader.K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
        coord_vis = draw_xyz_axis(color.copy(), ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        
        # Get mesh vertices and faces
        vertices = est.mesh.vertices
        faces = est.mesh.faces

        # copy the vertices to the camera space
        vertices_world = vertices.copy()
        
        # Transform vertices to camera space
        vertices_homo = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
        vertices_cam = (pose @ vertices_homo.T).T[:, :3]

        camera_object_mesh = trimesh.Trimesh(vertices=vertices_cam, faces=faces)
        camera_object_meshes.append(camera_object_mesh)

        # Project vertices to image plane
        vertices_2d, _ = cv2.projectPoints(
            vertices_cam,
            np.zeros(3),  # rotation vector (zero since vertices are already in camera space)
            np.zeros(3),  # translation vector (zero since vertices are already in camera space)
            reader.K,
            None
        )
        vertices_2d = vertices_2d.reshape(-1, 2)
        # # Create a mask image with projected vertices as white
        # H, W = color.shape[:2]
        # mask_vis = np.zeros((H, W), dtype=np.uint8)  # Single channel mask
        # # Convert vertices_2d to integer coordinates for indexing
        # vertices_2d_int = vertices_2d.astype(np.int32)
        # # Create valid mask for coordinates within image bounds
        # valid_mask = (vertices_2d_int[:, 0] >= 0) & (vertices_2d_int[:, 0] < W) & \
        #             (vertices_2d_int[:, 1] >= 0) & (vertices_2d_int[:, 1] < H)
        # # Set valid projected vertices to white (255)
        # valid_vertices = vertices_2d_int[valid_mask]
        # mask_vis[valid_vertices[:, 1], valid_vertices[:, 0]] = 255
        # plt.imsave(os.path.join(temp_dir, f'mask_pred_vis_{i}.png'), mask_vis)
        # pdb.set_trace()
        # Save 2D mesh as OBJ (Z=0)
        # verts_2d_3d = np.hstack([vertices_2d, np.zeros((vertices_2d.shape[0], 1))])
        # mesh_2d = trimesh.Trimesh(vertices=verts_2d_3d, faces=faces)
        # temp_dir_2d = os.path.join(temp_dir, '2d_meshes')
        # print(f"temp_dir_2d {temp_dir_2d}")
        # os.makedirs(temp_dir_2d, exist_ok=True)
        # mesh_2d.export(os.path.join(temp_dir_2d, f'object_2d_mesh_{i}.obj'))
        # Create mask for visible faces
        mask = np.zeros(color.shape[:2], dtype=np.uint8)
        
        # Draw each face
        for face_idx, face in enumerate(faces):
            # Get 2D points for this face
            pts = vertices_2d[face].astype(np.int32)
            
            # Check if face is visible (all points are in front of camera)
            if np.all(vertices_cam[face, 2] > 0):
                # Draw filled polygon
                cv2.fillPoly(mask, [pts], 1)
        
        # Create visualization with just the mask
        mesh_vis = np.zeros_like(color)
        mesh_vis[mask > 0] = [255, 255, 255]  # White color for mask
        
        # save a image with the mesh_vis
        # output_path = os.path.join(temp_dir, f'mesh_vis_{i}.png')
        # plt.imsave(output_path, mesh_vis)

        # Compute mask error
        # Load target mask
        mask_file = os.path.join(reader.video_dir, 'masks_pred_obj', f'{reader.id_strs[i]}.png')
        if os.path.exists(mask_file):
            target_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            # save the target_mask
            # plt.imsave(os.path.join(temp_dir, f'target_mask_{i}.png'), target_mask)
            # Convert mesh_vis to grayscale mask (take the first channel since it's white)
            mesh_mask = (mesh_vis[:, :, 0] > 0).astype(np.uint8) * 255
            
            # Normalize both masks to [0, 1] range
            mesh_mask_norm = mesh_mask.astype(np.float32) / 255.0
            target_mask_norm = target_mask.astype(np.float32) / 255.0
            #pdb.set_trace()
            # Convert to PyTorch tensors
            mesh_mask_tensor = torch.from_numpy(mesh_mask_norm).float().to(device)
            target_mask_tensor = torch.from_numpy(target_mask_norm).float().to(device)
            
            # Compute multiple mask error metrics
            # 1. IoU (Intersection over Union) - better for mask comparison
            intersection = torch.logical_and(mesh_mask_tensor > 0.5, target_mask_tensor > 0.5).float().sum()
            # plot both the mesh_mask_tensor and target_mask_tensor in a single image
            combined_mask = np.zeros_like(mesh_mask)
            # Convert CUDA tensors to CPU before converting to numpy
            mesh_mask_cpu = mesh_mask_tensor.cpu().numpy()
            target_mask_cpu = target_mask_tensor.cpu().numpy()
            combined_mask[mesh_mask_cpu > 0.5] = 1
            combined_mask[target_mask_cpu > 0.5] = 2
            #plt.imsave(os.path.join(temp_dir, f'combined_mask_{i}.png'), combined_mask)
            # pdb.set_trace()
            union = torch.logical_or(mesh_mask_tensor > 0.5, target_mask_tensor > 0.5).float().sum()
            iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
            iou_error = 1.0 - iou  # Convert to error (lower is better)
            # pdb.set_trace()
            
            # 2. Dice coefficient (F1 score for binary masks)
            dice = (2 * intersection) / (mesh_mask_tensor.sum() + target_mask_tensor.sum() + 1e-8)
            dice_error = 1.0 - dice  # Convert to error (lower is better)
            
            # 3. Coverage error (how much of target is covered by prediction)
            coverage = intersection / (target_mask_tensor.sum() + 1e-8)
            coverage_error = 1.0 - coverage  # Convert to error (lower is better)
            
            # 4. Original MSE for comparison
            mse_error = torch.nn.functional.mse_loss(mesh_mask_tensor, target_mask_tensor)
            
            
            # Use weighted combination of IoU and shape metrics
            mask_error = 1.0 * iou_error
            rendering_error += mask_error.item()
            

            # Write down the mask error in a file
            with open(results_file, 'a') as f:
                f.write(f"Frame {i}: MSE={mse_error.item():.6f}, IoU={iou_error.item():.6f}, Dice={dice_error.item():.6f}, Coverage={coverage_error.item():.6f}\n")
            
            #output_path = os.path.join(temp_dir, f"mesh_vis_{i}.png")
            #plt.imsave(output_path, mesh_mask, cmap='gray')

        # Blend mask with original image
        alpha = 0.5
        blended = cv2.addWeighted(color, 1, mesh_vis, alpha, 0)
        # Add coordinate axes on top without blending
        coord_mask = (coord_vis != color).any(axis=2)
        blended[coord_mask] = coord_vis[coord_mask]
        # Write frame to video
        video_writer.write(blended[...,::-1])  # Convert RGB to BGR for OpenCV
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to: {video_path}")
    
    # Calculate average score
    avg_score = np.mean(scores)
    print(f"Scale {scale_factor}: Average score = {avg_score:.4f}")

    # Compute the rendering error
    # rendering_error = compute_rendering_error(resized_mesh, poses, reader, temp_dir)
    # rendering_error = 1.0
    print(f"Scale {scale_factor}: Rendering error = {rendering_error:.4f}")
    untransformed_vertices = est.mesh.vertices.copy()
    return {
        'scale_factor': scale_factor,
        'avg_score': avg_score,
        'scores': scores,
        'poses': poses,
        'mesh_path': mesh_path,
        'resized_mesh': resized_mesh,
        'video_path': video_path,
        'rendering_error': rendering_error,
        'meshes': camera_object_meshes,
        'untransformed_vertices': untransformed_vertices
    }


def main():
    # Set memory optimization settings
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocation settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--object_folder', type=str, default="cloth")
    parser.add_argument('--object_type', type=str, default="grasp")
    object_folder = parser.parse_args().object_folder
    object_type = parser.parse_args().object_type
    # Per-object working dirs. process_videos.sh / run_object_pos_est_in_docker.sh
    # forward DATA_ROOT; when running this script by hand inside the container we
    # fall back to FoundationPose's own demo_data/ layout.
    data_root = os.environ.get("DATA_ROOT") or os.path.join(code_dir, "demo_data")
    debug_root = os.environ.get("DEBUG_ROOT") or os.path.join(code_dir, "debug_multi_scale")
    parser.add_argument('--mesh_file', type=str, default=os.path.join(data_root, object_folder, 'mesh_original', f'textured_simple_{object_type}.obj'))
    parser.add_argument('--test_scene_dir', type=str, default=os.path.join(data_root, object_folder))
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default=os.path.join(debug_root, object_folder, object_type))
    parser.add_argument('--scale_range', type=str, default='0.6, 0.75, 0.9, 1.0, 1.1', help='Comma-separated scale factors to test') 
    # 0.6, 0.75, 0.9, 1.0, 0.9, 1.0, 1.1, 1.25,  , 0.75, 0.9, 1.0, 1.1, 1.25, 1.4 #, 1.0 0.9, 1.0, 1.1, 1.25
    # 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4 ,0.9, 1.0 , 1.1, 1.25, 1.4 0.9, 1.0, , 1.5 0.6, 0.75, 0.9, 1.0, 0.9, 1.0, 
    args = parser.parse_args()

    if not os.path.exists(args.mesh_file):
        # A clip may legitimately have no 'target' object; a missing 'grasp' mesh
        # means stage 6 (obj_mesh) has not been run for this object.
        if object_type == "target":
            print(f"No target-object mesh at {args.mesh_file}; skipping the target pass.")
            return 0
        print(f"ERROR: grasp-object mesh not found at {args.mesh_file}")
        print("Run the 'obj_mesh' stage first (reconstruction/process_videos.sh "
              "--stages obj_mesh), or place your own mesh there.")
        return 1
    set_logging_format()
    set_seed(0)
    # Load original mesh
    original_mesh_dir = os.path.dirname(args.mesh_file)
    mesh_basename = os.path.splitext(os.path.basename(args.mesh_file))[0]
    mesh = trimesh.load(args.mesh_file)
    
    # Ensure texture is loaded if texture files exist
    # Check if texture files exist and mesh doesn't have texture loaded
    if not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        # Try to find and load texture file
        tex_file = os.path.join(original_mesh_dir, f"{mesh_basename}.png")
        if os.path.exists(tex_file):
            print(f"Loading texture from: {tex_file}")
            im = Image.open(tex_file)
            # If mesh has UV coordinates, use them; otherwise unwrap
            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                mesh = mesh.unwrap()
            uv = mesh.visual.uv if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None else None
            if uv is not None:
                material = trimesh.visual.texture.SimpleMaterial(image=im)
                color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
                mesh.visual = color_visuals
                print(f"Texture loaded successfully from {tex_file}")
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        # Mesh already has texture, but ensure the image is loaded from file
        tex_file = os.path.join(original_mesh_dir, f"{mesh_basename}.png")
        if os.path.exists(tex_file):
            # Reload texture image to ensure it's fresh
            img = Image.open(tex_file)
            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                mesh.visual.material.image = img
                print(f"Updated texture image from {tex_file}")
    # print(f"Original mesh diameter: {mesh.bounds_diagonal:.4f}")
    # Parse scale factors
    scale_factors = [float(x.strip()) for x in args.scale_range.split(',')]
    print(f"Testing scale factors: {scale_factors}")

    # Initialize data reader
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf, object_type=object_type)
    # original_mesh_dir is already defined above when loading the mesh
    # Test each scale factor
    results = []
    mesh_dict = {}
    for scale_factor in tqdm(scale_factors, desc="Testing scales"):
        result = test_single_scale(
            original_mesh=mesh,
            scale_factor=scale_factor,
            reader=reader,
            debug_dir=args.debug_dir,
            est_refine_iter=args.est_refine_iter,
            track_refine_iter=args.track_refine_iter,
            original_mesh_dir=original_mesh_dir
        )
        results.append(result)
        mesh_dict[scale_factor] = {
            "mesh": result['meshes'], 
            "poses": result['poses'], 
            "untransformed_vertices": result['untransformed_vertices']}

    # Find best scale using rendering error (lower is better)
    valid_results = [r for r in results if r['avg_score'] >= 0]
    if not valid_results:
        print("No valid results found!")
        return

    best_result = min(valid_results, key=lambda x: x.get('rendering_error', float('inf')))
    
    print(f"\n=== RESULTS ===")
    for result in results:
        if result['avg_score'] >= 0:
            print(f"Scale {result['scale_factor']}: Pose Score = {result['avg_score']:.4f}, Rendering Error = {result.get('rendering_error', 'N/A'):.4f}")
        else:
            print(f"Scale {result['scale_factor']}: Failed")
    
    print(f"\n=== BEST SCALE (by rendering error) ===")
    print(f"Best scale factor: {best_result['scale_factor']}")
    print(f"Best rendering error: {best_result.get('rendering_error', 'N/A'):.4f}")
    print(f"Pose score: {best_result['avg_score']:.4f}")
    
    # Save the best mesh to the mesh directory
    best_mesh = best_result['resized_mesh']
    output_mesh_dir = os.path.join(args.test_scene_dir, 'mesh')
    os.makedirs(output_mesh_dir, exist_ok=True)
    
    output_mesh_path = os.path.join(output_mesh_dir, f'textured_simple_{object_type}.obj')
    best_mesh.export(output_mesh_path)
    print(f"Best mesh saved to: {output_mesh_path}")
    
    # Copy texture files if they exist
    original_mesh_dir = os.path.dirname(args.mesh_file)
    for file in os.listdir(original_mesh_dir):
        if file.endswith(('.mtl', '.png', '.jpg', '.jpeg')):
            src_path = os.path.join(original_mesh_dir, file)
            dst_path = os.path.join(output_mesh_dir, file)
            shutil.copy2(src_path, dst_path)
            print(f"Copied texture file: {file}")
    
    # Save results summary
    results_file = os.path.join(args.debug_dir, 'multi_scale_results.txt')
    with open(results_file, 'w') as f:
        f.write("Multi-Scale FoundationPose Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Object: {object_folder}\n") 
        # f.write(f"Original mesh diameter: {mesh.bounds_diagonal:.4f}\n\n")
        
        f.write("Scale Factor Results:\n")
        for result in results:
            if result['avg_score'] >= 0:
                f.write(f"  Scale {result['scale_factor']}: Pose Score = {result['avg_score']:.4f}, Rendering Error = {result.get('rendering_error', 'N/A'):.4f}\n")
                if result.get('video_path'):
                    f.write(f"    Video: {result['video_path']}\n")
            else:
                f.write(f"  Scale {result['scale_factor']}: Failed - {result.get('error', 'Unknown error')}\n")
        
        f.write(f"\nBest scale factor (by rendering error): {best_result['scale_factor']}\n")
        f.write(f"Best rendering error: {best_result.get('rendering_error', 'N/A'):.4f}\n")
        f.write(f"Pose score: {best_result['avg_score']:.4f}\n")
        f.write(f"Best mesh saved to: {output_mesh_path}\n")
        if best_result.get('video_path'):
            f.write(f"Best scale video: {best_result['video_path']}\n")
    
    print(f"\nResults summary saved to: {results_file}")
    print(f"Best mesh and textures saved to: {output_mesh_dir}")
    
    # # Print video locations
    # print(f"\nVideos saved for each scale:")
    # for result in results:
    #     if result.get('video_path'):
    #         print(f"  Scale {result['scale_factor']}: {result['video_path']}")

    # After best_result is found and you have per-frame camera meshes:
    # Same data root as the argument defaults above (DATA_ROOT when forwarded by
    # object_pos_est.bash / process_videos.sh, else FoundationPose's demo_data/).
    output_dir = os.path.join(data_root, object_folder, "obj_mesh")
    os.makedirs(output_dir, exist_ok=True)
    best_scale_factor = best_result['scale_factor']
    print(f"best_scale_factor {best_scale_factor} !!!")
    for frame_idx, mesh in enumerate(mesh_dict[best_scale_factor]['mesh']):
        # pose = mesh_info['poses']
        # untransformed_vertices = mesh_info['untransformed_vertices']
        # # transform the untransformed_vertices to the pose
        # transformed_vertices = (pose @ untransformed_vertices.T).T[:, :3]
        # mesh.vertices = transformed_vertices
        
        image_name = os.path.splitext(os.path.basename(reader.color_files[frame_idx]))[0]  # Remove extension
        mesh_path = os.path.join(output_dir, f"{image_name}_object_mesh_{object_type}.obj")
        mesh.export(mesh_path)
    mesh_pose_path = os.path.join(output_dir, f"object_mesh_poses_{object_type}.npy")
    np.save(mesh_pose_path, mesh_dict[best_scale_factor]['poses'])
    untransformed_vertices_path = os.path.join(output_dir, f"object_mesh_untransformed_vertices_{object_type}.npy")
    np.save(untransformed_vertices_path, mesh_dict[best_scale_factor]['untransformed_vertices'])

if __name__ == "__main__":
    raise SystemExit(main() or 0) 



# def compute_rendering_error(mesh, poses, reader, output_dir):
#     """
#     Compute rendering error using differential rendering.
    
#     Args:
#         mesh: Trimesh object
#         poses: List of poses (4x4 matrices)
#         reader: Data reader object
#         scale_factor: Scale factor used
    
#     Returns:
#         float: Average rendering error across frames
#     """
#     # Force CPU usage for PyTorch3D compatibility
#     device = torch.device("cuda:0")
    
#     # Convert trimesh to PyTorch3D mesh
#     vertices = torch.from_numpy(mesh.vertices).float().to(device)
#     faces = torch.from_numpy(mesh.faces).long().to(device)
    
#     # Create PyTorch3D mesh
#     pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces])
    
#     # Renderer setup with adjusted parameters to avoid overflow
#     raster_settings = RasterizationSettings(
#         image_size=(reader.get_color(0).shape[0], reader.get_color(0).shape[1]),
#         blur_radius=1e-6,
#         faces_per_pixel=10,
#         bin_size=0,  # Use naive rasterization to avoid overflow
#         max_faces_per_bin=100000,  # Increase max faces per bin
#     )
    
#     blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    
#     def get_renderer(R, T):
#         cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
#         renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
#             shader=SoftSilhouetteShader(blend_params=blend_params)
#         )
#         return renderer, cameras
    
#     total_error = 0.0
#     num_frames = min(len(reader.color_files), 20)   # Use first 5 frames for efficiency
#     results_file = os.path.join(output_dir, f"mask_error.txt")
#     for i in range(num_frames):
#         pose = poses[i]
#         # Extract rotation and translation from pose matrix
#         R = torch.from_numpy(pose[:3, :3]).float().to(device).unsqueeze(0)
#         T = torch.from_numpy(pose[:3, 3]).float().to(device).unsqueeze(0)
        
#         # Get renderer
#         renderer, cameras = get_renderer(R, T)
        
#         # Render mask
#         silhouette = renderer(pytorch3d_mesh)
#         pred_mask = silhouette[..., 3].squeeze()  # Shape: (H, W)
        
#         # Load target mask
#         mask_file = os.path.join(reader.video_dir, 'masks_pred', f'{reader.id_strs[i]}.png')
#         if os.path.exists(mask_file):
#             target_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
#             target_mask = torch.from_numpy(target_mask).float().to(device) / 255.0
            
#             # Resize target mask to match predicted mask
#             if target_mask.shape != pred_mask.shape:
#                 target_mask = torch.nn.functional.interpolate(
#                     target_mask.unsqueeze(0).unsqueeze(0), 
#                     size=pred_mask.shape, 
#                     mode='nearest'
#                 ).squeeze()
            
#             # Compute mask error
#             # pdb.set_trace()
#             mask_error = torch.nn.functional.mse_loss(pred_mask, target_mask)
#             total_error += mask_error.item()
#             # write down the mask error in a file
#             with open(results_file, 'w') as f:
#                 f.write(f"mask error in frame {i}: {mask_error}")
#             output_path = os.path.join(output_dir, f"pred_mask_{i}.png")
#             plt.imsave(output_path, pred_mask.cpu().detach().numpy(), cmap='gray')

#         else:
#             pdb.set_trace()
    
#     return total_error / num_frames if num_frames > 0 else 1.0