#!/usr/bin/env python3
"""
Test script for MANO parameter conversion functions.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hamer.models.hamer import HAMER
from yacs.config import CfgNode

def create_dummy_config():
    """Create a dummy configuration for testing."""
    cfg = CfgNode()
    
    # MANO configuration
    cfg.MANO = CfgNode()
    cfg.MANO.DATA_DIR = "_DATA/data/"
    cfg.MANO.MODEL_PATH = "_DATA/data/mano"
    cfg.MANO.GENDER = "neutral"
    cfg.MANO.NUM_HAND_JOINTS = 15
    cfg.MANO.MEAN_PARAMS = "_DATA/data/mano_mean_params.npz"
    cfg.MANO.CREATE_BODY_POSE = False
    
    # Model configuration
    cfg.MODEL = CfgNode()
    cfg.MODEL.BACKBONE = CfgNode()
    cfg.MODEL.BACKBONE.TYPE = "vit"
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
    
    cfg.MODEL.MANO_HEAD = CfgNode()
    cfg.MODEL.MANO_HEAD.TYPE = "transformer_decoder"
    cfg.MODEL.MANO_HEAD.IEF_ITERS = 1
    cfg.MODEL.MANO_HEAD.JOINT_REP = "6d"
    cfg.MODEL.MANO_HEAD.TRANSFORMER_INPUT = "zero"
    cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER = CfgNode()
    cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER.depth = 6
    cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER.heads = 8
    cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER.dropout = 0.1
    cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER.mlp_dim = 2048
    cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER.dim_head = 64
    
    # Training configuration
    cfg.TRAIN = CfgNode()
    cfg.TRAIN.LR = 0.001
    cfg.TRAIN.WEIGHT_DECAY = 0.0001
    
    # Loss weights
    cfg.LOSS_WEIGHTS = CfgNode()
    cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0
    
    return cfg

def test_mano_conversion():
    """Test the MANO parameter conversion functions."""
    
    # Load configuration
    cfg = create_dummy_config()
    
    # Create HAMER model
    model = HAMER(cfg, init_renderer=False)
    model.eval()
    
    # Create dummy MANO parameters (rotation matrices)
    batch_size = 2
    pred_mano_params = {
        'global_orient': torch.randn(batch_size, 3, 3),  # Random rotation matrices
        'hand_pose': torch.randn(batch_size, 15, 3, 3),  # Random rotation matrices for 15 joints
        'betas': torch.randn(batch_size, 10),  # Shape parameters
        'transl': torch.randn(batch_size, 3),   # Translation
    }
    
    # Normalize rotation matrices to be valid
    for key in ['global_orient', 'hand_pose']:
        matrices = pred_mano_params[key]
        # Use SVD to ensure valid rotation matrices
        U, _, V = torch.svd(matrices.view(-1, 3, 3))
        R = torch.matmul(U, V.transpose(-2, -1))
        # Ensure proper orientation
        det = torch.det(R)
        V_new = V.clone()
        V_new[:, :, 2] *= det.unsqueeze(-1)
        R = torch.matmul(U, V_new.transpose(-2, -1))
        pred_mano_params[key] = R.view(matrices.shape)
    
    print("Original MANO parameters:")
    print(f"  global_orient shape: {pred_mano_params['global_orient'].shape}")
    print(f"  hand_pose shape: {pred_mano_params['hand_pose'].shape}")
    
    # Convert to manotorch format
    manotorch_params = model.convert_mano_to_manotorch_params(pred_mano_params)
    print(f"\nConverted to manotorch format:")
    print(f"  manotorch_params shape: {manotorch_params.shape}")
    print(f"  Expected shape: ({batch_size}, 18)")
    
    # Convert back to MANO format
    converted_back = model.convert_manotorch_to_mano_params(manotorch_params)
    print(f"\nConverted back to MANO format:")
    print(f"  global_orient shape: {converted_back['global_orient'].shape}")
    print(f"  hand_pose shape: {converted_back['hand_pose'].shape}")
    
    # Check reconstruction error
    global_error = torch.norm(pred_mano_params['global_orient'] - converted_back['global_orient'], dim=(-2, -1)).mean()
    hand_error = torch.norm(pred_mano_params['hand_pose'] - converted_back['hand_pose'], dim=(-2, -1)).mean()
    
    print(f"\nReconstruction errors:")
    print(f"  Global orientation error: {global_error.item():.6f}")
    print(f"  Hand pose error: {hand_error.item():.6f}")
    
    # Test PCA components
    print(f"\nPCA components analysis:")
    print(f"  hands_mean shape: {model.hands_mean.shape}")
    print(f"  hands_components shape: {model.hands_components.shape}")
    print(f"  First 5 PCA components: {manotorch_params[0, 3:8]}")
    
    return manotorch_params, converted_back

if __name__ == "__main__":
    test_mano_conversion() 