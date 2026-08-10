# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name="contactopt",
    version="1.0.0",
    description="ContactOpt: Optimizing Contact to Improve Grasps",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # "torch",
        # "pytorch3d",
        "torch-geometric",
        "torch-cluster",  # Required for fps, radius functions
        "torch-scatter",  # Required for knn_interpolate and other operations
        "numpy",
        "tqdm",
        "open3d",
        "tensorboardX",
        "pyquaternion",
        "trimesh",
        "transforms3d",
        "chumpy",
        "opencv-python",
    ],
)

