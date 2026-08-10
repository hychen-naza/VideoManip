import numpy as np
import torch
from scipy.optimize import minimize
from typing import List, Tuple
import pdb

def umi_finger_retargeting(
    human_poses: np.ndarray,  # Shape: (21, 3) - MANO joint positions
    human_joints: np.ndarray,  # Shape: (21, 3) - MANO joint positions
    umi_hand_model,
    target_human_indices: List[int] = [4, 8],  # Human hand joint indices for left and right fingers
    target_umi_links: List[str] = ["left_finger", "right_finger"]  # UMI gripper finger links
):
    """
    Custom retargeting for UMI gripper that maps human hand joints to gripper fingers.
    
    Args:
        human_poses: MANO joint positions (21, 3)
        human_joints: MANO joint positions (21, 3)
        umi_hand_model: UMI hand model for forward kinematics
        target_human_indices: Human joint indices to target (default: [4, 8] for index and pinky tips)
        target_umi_links: UMI gripper finger links to target (default: ["left_finger", "right_finger"])
    
    Returns:
        qpos: Optimized joint positions for UMI gripper (8 DOF: 6 base + 2 finger joints)
    """
    
    # Extract target human joint positions
    target_human_positions = human_joints[target_human_indices]  # Shape: (2, 3)
    
    def objective_function(qpos):
        """Objective function to minimize distance between human and UMI finger positions"""
        # Set the gripper to the current qpos
        # pdb.set_trace()
        umi_hand_model.set_qpos(torch.tensor(qpos, dtype=torch.float32))
        
        # Get current UMI finger positions
        umi_finger_positions = []
        for link_name in target_umi_links:
            link_pose = umi_hand_model.get_link_pose(link_name)
            umi_finger_positions.append(link_pose[:3, 3].numpy())  # Extract position
        
        umi_finger_positions = np.array(umi_finger_positions)  # Shape: (2, 3)
        
        # Calculate distances between human and UMI finger positions
        distances = np.linalg.norm(target_human_positions - umi_finger_positions, axis=1)
        # print(f"umi_finger_positions: {umi_finger_positions}")
        # print(f"target_human_positions: {target_human_positions}")
        # print(f"distances: {distances}")
        # Return sum of squared distances
        return np.sum(distances ** 2)
    
    # Initial guess for qpos (8 DOF: 6 base + 2 finger joints)
    # Base position and orientation + finger joint angles
    initial_qpos = np.array([human_joints[0][0], human_joints[0][1], human_joints[0][2], human_poses[0], human_poses[1], np.pi, -0.0, 0.0])  # Center position, fingers slightly open
    
    # Define bounds for optimization
    # Base position bounds (x, y, z, roll, pitch, yaw)
    base_bounds = [
        # (-0.5, 0.5),   # x translation
        # (-0.5, 0.5),   # y translation  
        # (-0.5, 0.5),   # z translation
        (-1., 1.),   # x translation
        (-1., 1.),   # y translation  
        (-1., 1.),   # z translation
        (-np.pi, np.pi),  # roll
        (-np.pi, np.pi),  # pitch
        (-np.pi, np.pi),  # yaw
    ]
    
    # Finger joint bounds (left_finger_joint, right_finger_joint)
    finger_bounds = [
        (-0.05, 0),   # left_finger_joint (0 to 5cm)
        (0.0, 0.05),   # right_finger_joint (0 to 5cm)
    ]
    
    all_bounds = base_bounds + finger_bounds
    
    # Optimize qpos to minimize distance between human and UMI finger positions
    result = minimize(
        objective_function,
        initial_qpos,
        method='L-BFGS-B',
        bounds=all_bounds,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )

    # print the result error
    print(f"Result error: {result.fun}")
    # pdb.set_trace()
    if not result.success:
        print(f"Warning: Optimization failed to converge. Success: {result.success}")
        print(f"Message: {result.message}")
    
    optimized_qpos = result.x
    
    return optimized_qpos, result.fun

def umi_retarget_with_constraints(
    human_joints: np.ndarray,
    umi_hand_model,
    target_human_indices: List[int] = [4, 8],
    target_umi_links: List[str] = ["left_finger", "right_finger"],
    wrist_position: np.ndarray = None,
    wrist_orientation: np.ndarray = None
):
    """
    Enhanced retargeting with additional constraints for wrist position and orientation.
    
    Args:
        human_joints: MANO joint positions (21, 3)
        umi_hand_model: UMI hand model
        target_human_indices: Human joint indices to target
        target_umi_links: UMI gripper finger links to target
        wrist_position: Optional wrist position constraint (3,)
        wrist_orientation: Optional wrist orientation constraint (4,) quaternion
    
    Returns:
        qpos: Optimized joint positions for UMI gripper
    """
    
    # Extract target human joint positions
    target_human_positions = human_joints[target_human_indices]
    
    def objective_function(qpos):
        """Objective function with wrist constraints"""
        umi_hand_model.set_qpos(torch.tensor(qpos, dtype=torch.float32))
        
        # Get UMI finger positions
        umi_finger_positions = []
        for link_name in target_umi_links:
            link_pose = umi_hand_model.get_link_pose(link_name)
            umi_finger_positions.append(link_pose[:3, 3].numpy())
        
        umi_finger_positions = np.array(umi_finger_positions)
        
        # Finger position error
        finger_error = np.sum(np.linalg.norm(target_human_positions - umi_finger_positions, axis=1) ** 2)
        
        # Wrist position error (if provided)
        wrist_error = 0.0
        if wrist_position is not None:
            base_pose = umi_hand_model.get_link_pose("base_link")
            base_position = base_pose[:3, 3].numpy()
            wrist_error = np.sum((base_position - wrist_position) ** 2)
        
        # Wrist orientation error (if provided)
        orientation_error = 0.0
        if wrist_orientation is not None:
            base_pose = umi_hand_model.get_link_pose("base_link")
            base_orientation = base_pose[:3, :3].numpy()
            # Convert to quaternion and compute error
            # This is a simplified orientation error
            orientation_error = 0.1  # Placeholder
        
        # Weighted sum of errors
        total_error = finger_error + 0.1 * wrist_error + 0.05 * orientation_error
        
        return total_error
    
    # Initial guess
    initial_qpos = np.array([0.0, 0.0, 0.0, 0.0, -np.pi/2, np.pi, 0.0, 0.0])
    
    # Bounds
    base_bounds = [
        (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),  # x, y, z
        (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),  # roll, pitch, yaw
    ]
    finger_bounds = [(0.0, 0.05), (0.0, 0.05)]  # left, right finger joints
    all_bounds = base_bounds + finger_bounds
    
    # Optimize
    result = minimize(
        objective_function,
        initial_qpos,
        method='L-BFGS-B',
        bounds=all_bounds,
        options={'maxiter': 150, 'ftol': 1e-6}
    )
    
    # print the result error
    print(f"Result error: {result.fun}")
    if not result.success:
        print(f"Warning: Optimization failed. Success: {result.success}, Message: {result.message}")
    
    return result.x 