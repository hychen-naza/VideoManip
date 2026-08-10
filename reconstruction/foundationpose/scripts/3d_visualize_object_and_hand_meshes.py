#!/usr/bin/env python3
"""
Script to visualize the object mesh and human hand mesh of each frame in the same coordinate system.
"""

import os
import trimesh
import glob
import argparse
import numpy as np
import pyrender
import pdb
import cv2
# from farthest_point_sampling import farthest_point_sampling
def farthest_point_sampling(point_cloud, num_points=1024):
    """
    :param point_cloud: (N, 3) or (N, 4), point cloud (with link index) - numpy array
    :param num_points: int, number of sampled points
    :return: ((N, 3) or (N, 4), list), sampled point cloud (numpy) & index
    """
    point_cloud_origin = point_cloud
    if point_cloud.shape[1] == 4:
        point_cloud = point_cloud[:, :3]

    selected_indices = [0]
    distances = np.linalg.norm(point_cloud - point_cloud[selected_indices[-1]], axis=1)
    for _ in range(num_points - 1):
        farthest_point_idx = np.argmax(distances)
        selected_indices.append(farthest_point_idx)
        new_distances = np.linalg.norm(point_cloud - point_cloud[farthest_point_idx], axis=1)
        distances = np.minimum(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[selected_indices]

    return sampled_point_cloud, selected_indices

def load_meshes(rgb_dir, hand_dir, object_dir, object_suffix='_object_mesh.obj', hand_suffix='_hand_mesh.obj'):
    """
    Load object and hand meshes for each frame from the directory.
    Returns a list of (object_mesh, hand_mesh, image_name) tuples.
    """
    output_dir = os.path.join(os.path.dirname(object_dir), "scene")
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    for i, rgb_file in enumerate(rgb_files):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        object_file = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
        # object_file = os.path.join(object_dir, f"object_2d_mesh_{i}.obj") # TODO: change to the correct object file
        # Search for any file in hand_dir that starts with {image_name}_right and ends with .obj
        hand_candidates = glob.glob(os.path.join(hand_dir, f"{image_name}_right_3d_*.obj"))
        # hand_candidates = glob.glob(os.path.join(hand_dir, f"{image_name}_optimized_hand_mesh.obj"))
        # pdb.set_trace()
        if os.path.exists(object_file) and hand_candidates:
            obj_mesh = trimesh.load(object_file, force='mesh')
            # save the obj_mesh to the output_dir
            scene = trimesh.Scene()
            scene.add_geometry(obj_mesh)
            # save the hand_candidates to the output_dir
            for hand_file in hand_candidates:
                hand_mesh = trimesh.load(hand_file, force='mesh')
                scene.add_geometry(hand_mesh)
                # pdb.set_trace()
            print(f"output_dir {output_dir}")
            # visualize the scene
            # scene.show()
            scene.export(os.path.join(output_dir, f"{image_name}_scene.obj"))

        if i > 1:
            break

def load_meshes_and_project(rgb_dir, hand_dir, object_dir, object_suffix='_object_mesh.obj', hand_suffix='_hand_mesh.obj'):
    """
    Load object and hand meshes for each frame from the directory and project them onto RGB images.
    """
    output_dir = os.path.join(os.path.dirname(object_dir), "projected_images")
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    for i, rgb_file in enumerate(rgb_files):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        object_file = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
        # Search for any file in hand_dir that starts with {image_name}_right and ends with .obj
        hand_candidates = glob.glob(os.path.join(hand_dir, f"{image_name}_right_3d_*.obj"))
        print(f"object_file {object_file}, hand_candidates {hand_candidates}")
        if os.path.exists(object_file) and hand_candidates:
            # Load RGB image
            rgb_image = cv2.imread(rgb_file)
            if rgb_image is None:
                print(f"Could not load image: {rgb_file}")
                continue
                
            # Load object mesh and project to 2D
            obj_mesh = trimesh.load(object_file, force='mesh')
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
            
            # Load and project hand meshes
            for hand_file in hand_candidates:
                hand_mesh = trimesh.load(hand_file, force='mesh')
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
            
            # Save the projected image
            output_path = os.path.join(output_dir, f"{image_name}_projected.png")
            cv2.imwrite(output_path, rgb_image)
            print(f"Saved projected image: {output_path}")

        # if i > 1:
        #     break

def load_meshes_and_project_video(rgb_dir, hand_dir, object_dir, object_suffix='_object_mesh.obj', hand_suffix='_hand_mesh.obj'):
    """
    Load object and hand meshes for each frame from the directory and project them onto RGB images.
    """
    output_dir = os.path.join(os.path.dirname(object_dir), "projected_images")
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    for i, rgb_file in enumerate(rgb_files):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        object_file = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
        # Search for any file in hand_dir that starts with {image_name}_right and ends with .obj
        hand_candidates = glob.glob(os.path.join(hand_dir, f"{image_name}_right_3d_*.obj"))
        
        if os.path.exists(object_file) and hand_candidates:
            # Load RGB image
            rgb_image = cv2.imread(rgb_file)
            if rgb_image is None:
                print(f"Could not load image: {rgb_file}")
                continue
                
            # Load object mesh and project to 2D
            obj_mesh = trimesh.load(object_file, force='mesh')
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
            
            # Load and project hand meshes
            for hand_file in hand_candidates:
                hand_mesh = trimesh.load(hand_file, force='mesh')
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
            



def visualize_3d_interactions(rgb_dir, hand_dir, object_dir, object_suffix='_object_mesh.obj', hand_suffix='_hand_mesh.obj'):
    """
    Visualize hand-object interactions in 3D using trimesh.
    """
    output_dir = os.path.join(os.path.dirname(object_dir), "3d_interactions")
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
    K = np.loadtxt(K_path)
    
    for i, rgb_file in enumerate(rgb_files):
        image_name = os.path.basename(rgb_file).replace(".png", "")
        object_file = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
        hand_candidates = glob.glob(os.path.join(hand_dir, f"{image_name}_right_3d_*.obj"))
        
        if os.path.exists(object_file) and hand_candidates:
            # Load object mesh
            obj_mesh = trimesh.load(object_file, force='mesh')
            obj_vertices_2d, _ = cv2.projectPoints(
                obj_mesh.vertices,
                np.zeros(3),
                np.zeros(3),
                K,
                None
            )
            obj_vertices_2d = obj_vertices_2d.reshape(-1, 2)
            
            # Sample furthest vertices from object using farthest point sampling (500 vertices)
            obj_vertices_3d_sampled, obj_furthest_indices = farthest_point_sampling(obj_mesh.vertices, 500)
            obj_vertices_2d_sampled = obj_vertices_2d[obj_furthest_indices]
            
            # Find the hand mesh that is closest to the object
            closest_hand_file = None
            min_avg_distance = float('inf')
            
            # First pass: find the closest hand to the object
            for hand_file in hand_candidates:
                hand_mesh = trimesh.load(hand_file, force='mesh')
                hand_vertices_2d, _ = cv2.projectPoints(
                    hand_mesh.vertices,
                    np.zeros(3),
                    np.zeros(3),
                    K,
                    None
                )
                hand_vertices_2d = hand_vertices_2d.reshape(-1, 2)
                
                # Calculate average distance from hand to object center
                hand_center_2d = np.mean(hand_vertices_2d, axis=0)
                obj_center_2d = np.mean(obj_vertices_2d, axis=0)
                avg_distance = np.linalg.norm(hand_center_2d - obj_center_2d)
                
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    closest_hand_file = hand_file
            
            # Second pass: only process the closest hand
            if closest_hand_file:
                hand_mesh = trimesh.load(closest_hand_file, force='mesh')
                hand_vertices_2d, _ = cv2.projectPoints(
                    hand_mesh.vertices,
                    np.zeros(3),
                    np.zeros(3),
                    K,
                    None
                )
                hand_vertices_2d = hand_vertices_2d.reshape(-1, 2)
                
                # Sample furthest vertices from hand using farthest point sampling (500 vertices)
                hand_vertices_3d_sampled, hand_furthest_indices = farthest_point_sampling(hand_mesh.vertices, 500)
                hand_vertices_2d_sampled = hand_vertices_2d[hand_furthest_indices]
                
                # Find close vertices between sampled hand and object vertices
                threshold_distance = 10.0  # pixels - adjust as needed
                interaction_pairs = []
                
                for hand_idx, hand_vertex_2d in enumerate(hand_vertices_2d_sampled):
                    for obj_idx, obj_vertex_2d in enumerate(obj_vertices_2d_sampled):
                        # Calculate 2D distance
                        distance_2d = np.linalg.norm(hand_vertex_2d - obj_vertex_2d)
                        
                        if distance_2d < threshold_distance:
                            # Get corresponding 3D vertices
                            hand_vertex_3d = hand_vertices_3d_sampled[hand_idx]
                            obj_vertex_3d = obj_vertices_3d_sampled[obj_idx]
                            
                            # Calculate 3D distance
                            distance_3d = np.linalg.norm(hand_vertex_3d - obj_vertex_3d)
                            
                            interaction_pairs.append({
                                'hand_vertex_3d': hand_vertex_3d,
                                'object_vertex_3d': obj_vertex_3d,
                                'distance_3d': distance_3d,
                                'hand_idx': hand_furthest_indices[hand_idx],
                                'object_idx': obj_furthest_indices[obj_idx],
                                'hand_file': os.path.basename(closest_hand_file)
                            })
                
                # Create 3D visualization
                if interaction_pairs:
                    # Create a scene
                    scene = trimesh.Scene()
                    
                    # Add original meshes with transparency
                    obj_mesh.visual.face_colors = [100, 150, 100, 100]  # Semi-transparent green
                    hand_mesh.visual.face_colors = [150, 100, 100, 100]  # Semi-transparent red
                    
                    scene.add_geometry(obj_mesh)
                    scene.add_geometry(hand_mesh)
                    
                    # Add interaction points and lines
                    for pair in interaction_pairs:
                        hand_vertex_3d = pair['hand_vertex_3d']
                        obj_vertex_3d = pair['object_vertex_3d']

                        hand_idx = pair['hand_idx']
                        object_idx = pair['object_idx']
                        
                        # Create spheres for interaction points
                        hand_sphere = trimesh.creation.icosphere(radius=0.005)
                        hand_sphere.visual.face_colors = [255, 0, 0, 255]  # Red for hand interaction points
                        hand_sphere.apply_translation(hand_vertex_3d)
                        
                        obj_sphere = trimesh.creation.icosphere(radius=0.005)
                        obj_sphere.visual.face_colors = [0, 255, 0, 255]  # Green for object interaction points
                        obj_sphere.apply_translation(obj_vertex_3d)
                        
                        scene.add_geometry(hand_sphere)
                        scene.add_geometry(obj_sphere)
                        
                        # Create line between interaction points using a thin cylinder
                        line_length = np.linalg.norm(obj_vertex_3d - hand_vertex_3d)
                        if line_length > 0.001:  # Only create line if points are not too close
                            # Create a thin cylinder to represent the line
                            line_cylinder = trimesh.creation.cylinder(radius=0.001, height=line_length, segment=[hand_vertex_3d, obj_vertex_3d])
                            line_cylinder.visual.face_colors = [255, 255, 0, 255]  # Yellow for interaction lines
                            scene.add_geometry(line_cylinder)
                    
                    # Save the 3D scene
                    scene_path = os.path.join(output_dir, f"{image_name}_3d_interactions.obj")
                    scene.export(scene_path)
                    
                    # Also save as GLB for better visualization
                    glb_path = os.path.join(output_dir, f"{image_name}_3d_interactions.glb")
                    scene.export(glb_path)
                    
                    print(f"Found {len(interaction_pairs)} 3D interactions in {image_name}")
                    print(f"Saved 3D visualization to: {scene_path}")
                    print(f"Saved GLB file to: {glb_path}")
                else:
                    print(f"No 3D interactions found in {image_name}")

        if i > 1:
            break


def main():
    parser = argparse.ArgumentParser(description='Visualize object and hand meshes per frame')
    parser.add_argument('--object_dir', type=str, required=True, help='Directory containing *_object_mesh.obj and *_hand_mesh.obj files')
    parser.add_argument('--mode', type=str, default='project', choices=['3d_vis', 'project', 'interaction', '3d_interaction'], 
                       help='Mode: "project" for simple projection, "interaction" for finding hand-object interactions, "3d_interaction" for 3D visualization')
    args = parser.parse_args()

    rgb_dir = os.path.join(args.object_dir, "rgb")
    hand_dir = os.path.join(args.object_dir, "human_hand")
    object_dir = os.path.join(args.object_dir, "obj_mesh")

    if args.mode == 'project':
        load_meshes_and_project(rgb_dir, hand_dir, object_dir)
    elif args.mode == '3d_vis':
        load_meshes(rgb_dir, hand_dir, object_dir)
    # elif args.mode == 'interaction':
    #     vertices_interaction_2d3d(rgb_dir, hand_dir, object_dir)
    elif args.mode == '3d_vis':
        load_meshes(rgb_dir, hand_dir, object_dir)
    elif args.mode == '3d_interaction':
        visualize_3d_interactions(rgb_dir, hand_dir, object_dir)


if __name__ == "__main__":
    main() 



# def vertices_interaction_2d3d(rgb_dir, hand_dir, object_dir, object_suffix='_object_mesh.obj', hand_suffix='_hand_mesh.obj'):
#     """
#     Find close 2D vertices between hand and object interactions and map them back to 3D.
#     """
#     output_dir = os.path.join(os.path.dirname(object_dir), "interaction_vertices")
#     os.makedirs(output_dir, exist_ok=True)
#     rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
#     K_path = os.path.join(os.path.dirname(rgb_dir), "cam_K.txt")
#     K = np.loadtxt(K_path)
    
#     for i, rgb_file in enumerate(rgb_files):
#         image_name = os.path.basename(rgb_file).replace(".png", "")
#         object_file = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
#         hand_candidates = glob.glob(os.path.join(hand_dir, f"{image_name}_right_3d_*.obj"))
        
#         if os.path.exists(object_file) and hand_candidates:
#             # Load RGB image
#             rgb_image = cv2.imread(rgb_file)
#             if rgb_image is None:
#                 print(f"Could not load image: {rgb_file}")
#                 continue
                
#             # Load object mesh and project to 2D
#             obj_mesh = trimesh.load(object_file, force='mesh')
#             obj_vertices_2d, _ = cv2.projectPoints(
#                 obj_mesh.vertices,
#                 np.zeros(3),
#                 np.zeros(3),
#                 K,
#                 None
#             )
#             obj_vertices_2d = obj_vertices_2d.reshape(-1, 2)
            
#             # Sample furthest vertices from object using farthest point sampling (500 vertices)
#             obj_vertices_3d_sampled, obj_furthest_indices = farthest_point_sampling(obj_mesh.vertices, 500)
#             obj_vertices_2d_sampled = obj_vertices_2d[obj_furthest_indices]
            
#             # Store interaction pairs
#             interaction_pairs = []
            
#             # Find the hand mesh that is closest to the object
#             closest_hand_file = None
#             min_avg_distance = float('inf')
            
#             # First pass: find the closest hand to the object
#             for hand_file in hand_candidates:
#                 hand_mesh = trimesh.load(hand_file, force='mesh')
#                 hand_vertices_2d, _ = cv2.projectPoints(
#                     hand_mesh.vertices,
#                     np.zeros(3),
#                     np.zeros(3),
#                     K,
#                     None
#                 )
#                 hand_vertices_2d = hand_vertices_2d.reshape(-1, 2)
                
#                 # Calculate average distance from hand to object center
#                 hand_center_2d = np.mean(hand_vertices_2d, axis=0)
#                 obj_center_2d = np.mean(obj_vertices_2d, axis=0)
#                 avg_distance = np.linalg.norm(hand_center_2d - obj_center_2d)
                
#                 if avg_distance < min_avg_distance:
#                     min_avg_distance = avg_distance
#                     closest_hand_file = hand_file
            
#             # Second pass: only process the closest hand
#             if closest_hand_file:
#                 hand_mesh = trimesh.load(closest_hand_file, force='mesh')
#                 hand_vertices_2d, _ = cv2.projectPoints(
#                     hand_mesh.vertices,
#                     np.zeros(3),
#                     np.zeros(3),
#                     K,
#                     None
#                 )
#                 hand_vertices_2d = hand_vertices_2d.reshape(-1, 2)
                
#                 # Sample furthest vertices from hand using farthest point sampling (500 vertices)
#                 hand_vertices_3d_sampled, hand_furthest_indices = farthest_point_sampling(hand_mesh.vertices, 500)
#                 hand_vertices_2d_sampled = hand_vertices_2d[hand_furthest_indices]
                
#                 # Find close vertices between sampled hand and object vertices
#                 threshold_distance = 10.0  # pixels - adjust as needed
                
#                 for hand_idx, hand_vertex_2d in enumerate(hand_vertices_2d_sampled):
#                     for obj_idx, obj_vertex_2d in enumerate(obj_vertices_2d_sampled):
#                         # Calculate 2D distance
#                         distance_2d = np.linalg.norm(hand_vertex_2d - obj_vertex_2d)
                        
#                         if distance_2d < threshold_distance:
#                             # Get corresponding 3D vertices
#                             hand_vertex_3d = hand_vertices_3d_sampled[hand_idx]
#                             obj_vertex_3d = obj_vertices_3d_sampled[obj_idx]
                            
#                             # Calculate 3D distance
#                             distance_3d = np.linalg.norm(hand_vertex_3d - obj_vertex_3d)
                            
#                             interaction_pairs.append({
#                                 'hand_vertex_2d': hand_vertex_2d,
#                                 'object_vertex_2d': obj_vertex_2d,
#                                 'hand_vertex_3d': hand_vertex_3d,
#                                 'object_vertex_3d': obj_vertex_3d,
#                                 'distance_2d': distance_2d,
#                                 'distance_3d': distance_3d,
#                                 'hand_idx': hand_furthest_indices[hand_idx],  # Original index
#                                 'object_idx': obj_furthest_indices[obj_idx],   # Original index
#                                 'hand_file': os.path.basename(closest_hand_file)
#                             })
                
#                 # Draw all original hand vertices (light blue, small dots)
#                 for hand_vertex_2d in hand_vertices_2d:
#                     x, y = int(hand_vertex_2d[0]), int(hand_vertex_2d[1])
#                     if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
#                         cv2.circle(rgb_image, (x, y), 1, (255, 255, 0), -1)  # Light blue for all hand vertices
                
#                 # Draw all original object vertices (light green, small dots)
#                 for obj_vertex_2d in obj_vertices_2d:
#                     x, y = int(obj_vertex_2d[0]), int(obj_vertex_2d[1])
#                     if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
#                         cv2.circle(rgb_image, (x, y), 1, (0, 255, 255), -1)  # Light green for all object vertices
                
#                 # Draw sampled vertices (larger dots)
#                 for hand_vertex_2d in hand_vertices_2d_sampled:
#                     x, y = int(hand_vertex_2d[0]), int(hand_vertex_2d[1])
#                     if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
#                         cv2.circle(rgb_image, (x, y), 3, (255, 0, 0), -1)  # Blue for sampled hand vertices
                
#                 for obj_vertex_2d in obj_vertices_2d_sampled:
#                     x, y = int(obj_vertex_2d[0]), int(obj_vertex_2d[1])
#                     if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
#                         cv2.circle(rgb_image, (x, y), 3, (0, 255, 0), -1)  # Green for sampled object vertices
                
#                 # Draw interaction points on image
#                 for pair in interaction_pairs:
#                     # Draw hand vertex (red)
#                     hand_x, hand_y = int(pair['hand_vertex_2d'][0]), int(pair['hand_vertex_2d'][1])
#                     if 0 <= hand_x < rgb_image.shape[1] and 0 <= hand_y < rgb_image.shape[0]:
#                         cv2.circle(rgb_image, (hand_x, hand_y), 6, (0, 0, 255), -1)
                    
#                     # Draw object vertex (red)
#                     obj_x, obj_y = int(pair['object_vertex_2d'][0]), int(pair['object_vertex_2d'][1])
#                     if 0 <= obj_x < rgb_image.shape[1] and 0 <= obj_y < rgb_image.shape[0]:
#                         cv2.circle(rgb_image, (obj_x, obj_y), 6, (0, 0, 255), -1)
                    
#                     # Draw line between interacting vertices (yellow)
#                     cv2.line(rgb_image, (hand_x, hand_y), (obj_x, obj_y), (0, 255, 255), 3)
            
#             # Save the interaction visualization
#             output_path = os.path.join(output_dir, f"{image_name}_interactions.png")
#             cv2.imwrite(output_path, rgb_image)
            
#             # Save interaction data
#             if interaction_pairs:
#                 interaction_data = {
#                     'image_name': image_name,
#                     'interaction_pairs': interaction_pairs,
#                     'total_interactions': len(interaction_pairs)
#                 }
                
#                 # Save as numpy file for easy loading
#                 np.save(os.path.join(output_dir, f"{image_name}_interactions.npy"), interaction_data)
                
#                 print(f"Found {len(interaction_pairs)} interactions in {image_name}")
#                 print(f"Saved interaction data to: {output_dir}")
#             else:
#                 print(f"No interactions found in {image_name}")

#         if i > 1:
#             break

