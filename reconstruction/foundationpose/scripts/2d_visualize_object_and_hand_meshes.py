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

def draw_mesh_on_image(image, mesh, color=(0, 255, 0), thickness=1):
    """
    Draws the mesh edges on the image.
    image: np.ndarray (H, W, 3)
    mesh: trimesh.Trimesh (with 2D vertices in mesh.vertices[:, :2])
    color: BGR tuple
    """
    img = image.copy()
    for face in mesh.faces:
        pts = mesh.vertices[face, :2].astype(np.int32)
        # Draw lines between each pair of vertices in the face
        for i in range(3):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i+1)%3])
            cv2.line(img, pt1, pt2, color, thickness)
    return img

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
        # object_file = os.path.join(object_dir, f"{image_name}_object_mesh.obj")
        object_file = os.path.join(object_dir, f"object_2d_mesh_{i}.obj") # TODO: change to the correct object file
        # Search for any file in hand_dir that starts with {image_name}_right and ends with .obj
        hand_candidates = glob.glob(os.path.join(hand_dir, f"{image_name}_right_2d_*.obj"))
        #pdb.set_trace()
        if os.path.exists(object_file) and hand_candidates:
            obj_mesh = trimesh.load(object_file, force='mesh')
            image = cv2.imread(rgb_file)
            overlay = draw_mesh_on_image(image, obj_mesh, color=(0, 0, 255), thickness=1)  # Red for object

            # Overlay all hand meshes
            for idx, hand_file in enumerate(hand_candidates):
                hand_mesh = trimesh.load(hand_file, force='mesh')
                overlay = draw_mesh_on_image(overlay, hand_mesh, color=(0, 255, 0), thickness=1)  # Green for hand

            # Save the overlay image
            overlay_dir = os.path.join(os.path.dirname(object_dir), "overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            cv2.imwrite(os.path.join(overlay_dir, f"{image_name}_overlay.png"), overlay)



def main():
    parser = argparse.ArgumentParser(description='Visualize object and hand meshes per frame')
    parser.add_argument('--object_dir', type=str, required=True, help='Directory containing *_object_mesh.obj and *_hand_mesh.obj files')
    args = parser.parse_args()

    rgb_dir = os.path.join(args.object_dir, "rgb")
    hand_dir = os.path.join(args.object_dir, "human_hand")
    object_dir = os.path.join(args.object_dir, "2d_meshes")

    load_meshes(rgb_dir, hand_dir, object_dir)


if __name__ == "__main__":
    main() 