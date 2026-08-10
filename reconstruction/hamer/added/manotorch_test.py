import torch
import os
from manotorch.manolayer import ManoLayer, MANOOutput
import pdb
import trimesh
# Manotorch should now find the assets in its own installation directory
print("Manotorch assets should be available in the conda environment installation")
print("Testing manotorch functionality...")
#mano_assets_path

# Select number of principal components for pose space
ncomps = 15

# initialize layers
mano_layer = ManoLayer(use_pca=True, flat_hand_mean=False, ncomps=ncomps)

batch_size = 2
# Generate random shape parameters
random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
random_pose = torch.rand(batch_size, 3 + ncomps)

# The mano_layer's output contains:
"""
MANOOutput = namedtuple(
    "MANOOutput",
    [
        "verts",
        "joints",
        "center_idx",
        "center_joint",
        "full_poses",
        "betas",
        "transforms_abs",
    ],
)
"""
# Test that manotorch can load and use the model
try:
    # forward mano layer
    pdb.set_trace()
    mano_output: MANOOutput = mano_layer(random_pose, random_shape)

    # retrieve 778 vertices, 21 joints and 16 SE3 transforms of each articulation
    # verts and joints in meters.
    verts = mano_output.verts  # (B, 778, 3), root(center_joint) relative
    joints = mano_output.joints  # (B, 21, 3), root relative
    transforms_abs = mano_output.transforms_abs  # (B, 16, 4, 4), root relative
    # how to get the faces and save the mesh to a file
    faces = mano_layer.th_faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export('manotorch_test.obj')
    pdb.set_trace()
    print("✓ Manotorch test successful!")
    print(f"Vertices shape: {verts.shape}")
    print(f"Joints shape: {joints.shape}")
    print(f"Transforms shape: {transforms_abs.shape}")
    
except Exception as e:
    print(f"✗ Error running manotorch: {e}")
    print("This might indicate an issue with the MANO model file or manotorch installation")