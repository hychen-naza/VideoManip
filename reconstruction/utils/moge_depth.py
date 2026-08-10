import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2
import pdb
import numpy as np
import os
from pathlib import Path
import argparse

def process_images_in_folder(folder_path, output_folder=None):
    """
    Process all images in a folder and save camera intrinsics for each.
    
    Args:
        folder_path (str): Path to folder containing images
        output_folder (str): Path to save camera intrinsics (default: camera_info under input folder)
    """
    device = torch.device("cuda")
    
    # Load the model from huggingface hub
    print("Loading MoGe model...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    
    # Set up input and output paths
    input_path = Path(folder_path)
    if output_folder is None:
        output_path = input_path / "camera_info"
    else:
        output_path = Path(output_folder)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.png']
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} image files:")
    for image_file in image_files:
        print(f"  - {image_file.name}")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
        
        # Read the input image and convert to tensor
        input_image = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        
        # Infer camera intrinsics
        output = model.infer(input_image)
        
        # Extract intrinsics
        H, W = input_image.shape[-2:]  # Height, Width
        intrinsics = output["intrinsics"].cpu().numpy().astype(np.float32)
        intrinsics[0, 0] *= W  # fx
        intrinsics[1, 1] *= H  # fy
        intrinsics[0, 2] *= W  # cx
        intrinsics[1, 2] *= H  # cy
        
        # Save intrinsics
        output_filename = f"camera_K_{image_file.name}.txt"
        output_file = output_path / output_filename
        np.savetxt(output_file, intrinsics)
        
        # Save depth
        depth_mm = (output['depth'].cpu().numpy() * 1000.0).astype(np.uint16)
        # Save as 16-bit PNG file (FoundationPose compatible)
        depth_output_file = output_path.parent / "depth" / image_file.name
        # create the depth folder if it doesn't exist
        os.makedirs(output_path.parent / "depth", exist_ok=True)
        cv2.imwrite(depth_output_file, depth_mm)

        print(f"Saved intrinsics to: {output_file} and depth to: {depth_output_file}")

        if (i == 0):
            output_filename = f"cam_K.txt"
            output_file = output_path.parent / output_filename
            np.savetxt(output_file, intrinsics)
    
    print(f"\nProcessing complete! Camera intrinsics saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract camera intrinsics from images using MoGe')
    parser.add_argument('folder_path', help='Path to folder containing images')
    parser.add_argument('--output_folder', help='Path to save camera intrinsics (default: camera_info under input folder)')
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.folder_path):
        print(f"Error: Input folder {args.folder_path} does not exist")
        return
    
    process_images_in_folder(args.folder_path, args.output_folder)

if __name__ == "__main__":
    # Example usage - you can modify this or use command line arguments
    # process_images_in_folder("/path/to/your/images")
    
    # For command line usage, uncomment the line below and comment out the example above
    main()
