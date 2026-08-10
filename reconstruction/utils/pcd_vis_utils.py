import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
import os
import cv2
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.cm as cm
from termcolor import cprint
import torch


def draw_pcd(pcd, title="Point Cloud", save_path=None, show=True, interactive=False):
    """
    Draw point cloud with different colors for hand and object points.
    
    Args:
        pcd: Point cloud tensor or numpy array of shape (N, 3) where N >= 1024
             First 512 points are hand, points 512:1024 are object
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    # Convert to numpy if tensor
    if torch.is_tensor(pcd):
        pcd = pcd.detach().cpu().numpy()
    
    # Handle batch dimension - take first sample if batch
    if len(pcd.shape) == 3:
        pcd = pcd[0]
    
    # Ensure we have at least 1024 points
    if pcd.shape[0] < 1024:
        raise ValueError(f"Point cloud must have at least 1024 points, got {pcd.shape[0]}")
    
    # Extract hand and object points
    hand_points = pcd[:512, :]  # First 512 points
    object_points = pcd[512:1024, :]  # Points 512:1024
    
    # Check if target object points exist
    has_target_object = len(pcd) > 1024
    if has_target_object:
        target_object_points = pcd[1024:, :]  # Points 1024:
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot hand points in blue
    ax.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], 
               c='blue', label='Hand', s=20, alpha=0.6)
    
    # Plot object points in orange/red
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], 
               c='orange', label='Object', s=20, alpha=0.6)
    
    # Plot target object points in red (if exists)
    if has_target_object:
        ax.scatter(target_object_points[:, 0], target_object_points[:, 1], target_object_points[:, 2], 
                   c='red', label='Target Object', s=20, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([pcd[:, 0].max() - pcd[:, 0].min(),
                          pcd[:, 1].max() - pcd[:, 1].min(),
                          pcd[:, 2].max() - pcd[:, 2].min()]).max() / 2.0
    mid_x = (pcd[:, 0].max() + pcd[:, 0].min()) * 0.5
    mid_y = (pcd[:, 1].max() + pcd[:, 1].min()) * 0.5
    mid_z = (pcd[:, 2].max() + pcd[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        # Check if user wants interactive HTML plot
        if interactive:
            # Save as interactive HTML using plotly
            fig_plotly = go.Figure()
            
            # Add hand points
            fig_plotly.add_trace(go.Scatter3d(
                x=hand_points[:, 0],
                y=hand_points[:, 1],
                z=hand_points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color='blue',
                    opacity=0.6
                ),
                name='Hand'
            ))
            
            # Add object points
            fig_plotly.add_trace(go.Scatter3d(
                x=object_points[:, 0],
                y=object_points[:, 1],
                z=object_points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color='orange',
                    opacity=0.6
                ),
                name='Object'
            ))
            
            # Add target object points if exists
            if has_target_object:
                fig_plotly.add_trace(go.Scatter3d(
                    x=target_object_points[:, 0],
                    y=target_object_points[:, 1],
                    z=target_object_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='red',
                        opacity=0.6
                    ),
                    name='Target Object'
                ))
            
            # Set layout
            fig_plotly.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='cube',
                    xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
                    yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
                    zaxis=dict(range=[mid_z - max_range, mid_z + max_range])
                ),
                width=1000,
                height=800
            )
            
            # Save as HTML
            # check if the file ends with .html
            if not save_path.endswith('.html'):
                # remove the last extension
                save_path = save_path.rsplit('.', 1)[0] + '.html'
            pio.write_html(fig_plotly, save_path)
            print(f"Interactive point cloud saved to {save_path}")
        else:
            # Save as static image
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Point cloud saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def draw_pcd_simple(robot_hand_vertices, object_vertices_grasp = None, title="Point Cloud", save_path=None, show=True, interactive=False):
    """
    Draw point cloud with different colors for robot hand and object points.
    
    Args:
        robot_hand_vertices: Robot hand point cloud tensor or numpy array of shape (N, 3)
        object_vertices_grasp: Optional object grasp point cloud tensor or numpy array of shape (M, 3)
        title: Title for the plot (not displayed, kept for compatibility)
        save_path: Optional path to save the figure
        show: Whether to display the plot
        interactive: Whether to save as interactive HTML (plotly)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(robot_hand_vertices):
        robot_hand_vertices = robot_hand_vertices.detach().cpu().numpy()

    # Process object vertices if provided
    if object_vertices_grasp is not None:
        if torch.is_tensor(object_vertices_grasp):
            object_vertices_grasp = object_vertices_grasp.detach().cpu().numpy()
        
        # Handle batch dimension
        if len(object_vertices_grasp.shape) == 3:
            object_vertices_grasp = object_vertices_grasp[0]
        
        # Ensure object_vertices_grasp is 2D
        if len(object_vertices_grasp.shape) != 2 or object_vertices_grasp.shape[1] != 3:
            raise ValueError(f"object_vertices_grasp must be shape (M, 3), got {object_vertices_grasp.shape}")
    
    # Combine all points for calculating bounds
    all_points = [robot_hand_vertices]
    if object_vertices_grasp is not None:
        all_points.append(object_vertices_grasp)
    all_points_combined = np.vstack(all_points)
    
    # Pink color from mesh_traj_render.py: [239, 132, 167] / 255.0
    pink_color = tuple(np.array([239, 132, 167], dtype=np.float32) / 255.0)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot robot hand points in pink
    ax.scatter(robot_hand_vertices[:, 0], robot_hand_vertices[:, 1], robot_hand_vertices[:, 2], 
               c='blue', s=20, alpha=0.6)
    
    # Plot object grasp points in pink (if provided)
    if object_vertices_grasp is not None:
        ax.scatter(object_vertices_grasp[:, 0], object_vertices_grasp[:, 1], object_vertices_grasp[:, 2], 
                   c=pink_color, s=20, alpha=0.6)
    
    # Completely remove/hide all axes
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Hide the panes (background grid planes)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # Hide grid lines
    ax.grid(False)
    # Hide axis spines (the lines forming the axes)
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Make tick labels invisible
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    
    # Set equal aspect ratio
    max_range = np.array([all_points_combined[:, 0].max() - all_points_combined[:, 0].min(),
                          all_points_combined[:, 1].max() - all_points_combined[:, 1].min(),
                          all_points_combined[:, 2].max() - all_points_combined[:, 2].min()]).max() / 2.0
    mid_x = (all_points_combined[:, 0].max() + all_points_combined[:, 0].min()) * 0.5
    mid_y = (all_points_combined[:, 1].max() + all_points_combined[:, 1].min()) * 0.5
    mid_z = (all_points_combined[:, 2].max() + all_points_combined[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        # Check if user wants interactive HTML plot
        if interactive:
            # Save as interactive HTML using plotly
            fig_plotly = go.Figure()
            
            # Pink color from mesh_traj_render.py: [239, 132, 167] / 255.0
            pink_color_rgb = f'rgb({239}, {132}, {167})'
            
            # Add robot hand points
            fig_plotly.add_trace(go.Scatter3d(
                x=robot_hand_vertices[:, 0],
                y=robot_hand_vertices[:, 1],
                z=robot_hand_vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=pink_color_rgb,
                    opacity=0.6
                ),
                showlegend=False
            ))
            
            # Add object grasp points if provided
            if object_vertices_grasp is not None:
                fig_plotly.add_trace(go.Scatter3d(
                    x=object_vertices_grasp[:, 0],
                    y=object_vertices_grasp[:, 1],
                    z=object_vertices_grasp[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=pink_color_rgb,
                        opacity=0.6
                    ),
                    showlegend=False
                ))
            
            # Set layout (no title, no axis labels)
            fig_plotly.update_layout(
                scene=dict(
                    xaxis=dict(
                        range=[mid_x - max_range, mid_x + max_range],
                        showbackground=False,
                        showticklabels=False,
                        title=""
                    ),
                    yaxis=dict(
                        range=[mid_y - max_range, mid_y + max_range],
                        showbackground=False,
                        showticklabels=False,
                        title=""
                    ),
                    zaxis=dict(
                        range=[mid_z - max_range, mid_z + max_range],
                        showbackground=False,
                        showticklabels=False,
                        title=""
                    ),
                    aspectmode='cube'
                ),
                width=1000,
                height=800,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            # Save as HTML
            # check if the file ends with .html
            if not save_path.endswith('.html'):
                # remove the last extension
                save_path = save_path.rsplit('.', 1)[0] + '.html'
            pio.write_html(fig_plotly, save_path)
            print(f"Interactive point cloud saved to {save_path}")
        else:
            # Save as static image
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Point cloud saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig