import os
import trimesh
import numpy as np
import imageio
import pyrender


def create_gif_and_glb(hand_path, object_path, output_prefix="output", num_frames=60, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # Load meshes using trimesh
    hand_trimesh = trimesh.load(hand_path, force='mesh')
    object_trimesh = trimesh.load(object_path, force='mesh')

    # Set up scene for GIF rendering
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[1.0, 1.0, 1.0])

    # Add hand mesh with material
    hand_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.4, 0.6, 0.8, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    hand_mesh = pyrender.Mesh.from_trimesh(hand_trimesh, material=hand_material, smooth=True)
    hand_node = scene.add(hand_mesh)

    # Add object mesh with material
    object_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.6, 0.7, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    object_mesh = pyrender.Mesh.from_trimesh(object_trimesh, material=object_material, smooth=True)
    object_node = scene.add(object_mesh)

    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],  # Move camera back slightly
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Offscreen renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    # Generate frames for GIF
    frames = []
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * np.pi
        rotation = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        scene.set_pose(hand_node, rotation)
        scene.set_pose(object_node, rotation)
        color, _ = renderer.render(scene)
        frames.append(color)

    gif_path = os.path.join(output_dir, f"{output_prefix}.gif")
    imageio.mimsave(gif_path, frames, duration=0.05, loop=0)
    print(f"[✓] Saved GIF: {gif_path}")

    # === EXPORT COMBINED GLB WITH COLOR ===

    # Apply color to hand mesh (light blue)
    hand_color = np.array([173, 216, 230, 255])  # light blue (RGB for #ADD8E6)
    hand_trimesh.visual.face_colors = np.tile(hand_color, (len(hand_trimesh.faces), 1))

    # Apply color to object mesh (light pink)
    object_color = np.array([255, 182, 193, 255])  # light pink (RGB for #FFB6C1)
    object_trimesh.visual.face_colors = np.tile(object_color, (len(object_trimesh.faces), 1))


    # Offset object mesh indices and concatenate both
    object_trimesh.apply_translation([0, 0, 0])  # Optional: position adjustment
    combined_mesh = trimesh.util.concatenate([hand_trimesh, object_trimesh])

    # Export GLB with color
    glb_path = os.path.join(output_dir, f"{output_prefix}.glb")
    combined_mesh.export(glb_path, file_type='glb')
    print(f"[✓] Saved GLB with color: {glb_path}")






if __name__ == "__main__":
    input_root = "/home/homanga/mesh/ours_sim_aug"
    output_root = "/home/homanga/mesh/output_mesh"

    for root, dirs, files in os.walk(input_root):
        if any(f.endswith("_hand.obj") for f in files):
            rel_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            for i in range(6):  # 0 to 5
                hand_file = os.path.join(root, f"{i}_hand.obj")
                object_file = os.path.join(root, f"{i}_object.obj")

                if os.path.exists(hand_file) and os.path.exists(object_file):
                    output_prefix = os.path.join(output_dir, str(i))
                    create_gif_and_glb(hand_file, object_file, output_prefix=output_prefix, output_dir=output_dir)