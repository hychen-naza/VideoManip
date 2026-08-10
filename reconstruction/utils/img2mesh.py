#!/usr/bin/env python3
"""
Image to Mesh Generator using Meshy API
Converts images to 3D meshes with texture and PBR materials.
"""

import os
import requests
import time
import json
import argparse
from pathlib import Path
import base64
from typing import Optional, Dict, Any
import logging
import pdb
from openai import OpenAI
import re
import trimesh
import numpy as np
import glob
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeshyImageToMesh:
    """Client for Meshy Image-to-3D API"""
    
    def __init__(self, api_key: str):
        """
        Initialize the Meshy client
        
        Args:
            api_key: Your Meshy API key
        """
        self.api_key = api_key
        self.base_url = "https://api.meshy.ai/openapi/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # openai client -- key comes from the environment, never from source
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. This stage calls the OpenAI API to "
                "complete the occluded object image before meshing. "
                "Export OPENAI_API_KEY and retry."
            )
        self.openai_client = OpenAI(api_key=openai_key)

    def generate_complete_object(self, image_paths: str) -> str:
        # object_name = self.query_image_object(image_paths)  # TODO: add object name
        result = self.openai_client.images.edit(
            model="gpt-image-1",
            image=[open(image_paths[i], "rb") for i in range(len(image_paths))],
            prompt=f"generata a image with complete object based on the provided images, only the object without hand or others" #, which is a bowl,
        )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Save the image to a file
        complete_image_path = image_paths[0].replace(".png", "_complete.png")
        print(f"Saving complete object to {complete_image_path}")
        with open(complete_image_path, "wb") as f:
            f.write(image_bytes)

        return complete_image_path

    def query_image_height(self, image_path: str) -> int:
        """
        Query the height of an image
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')

        response1 = self.openai_client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's the largest dimension—length, height, or width—of the object in this image? Please estimate it in meters"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }],
        )
        print(f"Response 1: {response1.output_text}")
        response_text = response1.output_text

        response2 = self.openai_client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Please summarize the object's largest dimension in meters in the format <value>m, based on the response: {response_text}"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }],
        )
        print(f"Response 2: {response2.output_text}")
        # extract the height from the response text
        # pdb.set_trace()
        height = re.search(r"(\d+\.\d+)", response2.output_text).group(1)
        return float(height)
    
    def submit_image_to_3d(self, 
                          image_path: str,
                          enable_pbr: bool = True,
                          should_remesh: bool = True,
                          should_texture: bool = True) -> str:
        """
        Submit an image for 3D mesh generation
        
        Args:
            image_path: Path to the input image
            enable_pbr: Enable PBR materials
            should_remesh: Enable mesh remeshing
            should_texture: Enable texture generation
            
        Returns:
            task_id: The task ID for tracking progress
        """
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image format
        image_ext = Path(image_path).suffix.lower()
        if image_ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif image_ext == '.png':
            mime_type = 'image/png'
        elif image_ext == '.webp':
            mime_type = 'image/webp'
        else:
            raise ValueError(f"Unsupported image format: {image_ext}")
        
        # Create data URI
        data_uri = f"data:{mime_type};base64,{base64_image}"
        
        # Prepare payload
        payload = {
            "image_url": data_uri,
            "enable_pbr": enable_pbr,
            "should_remesh": should_remesh,
            "should_texture": should_texture,
            "ai_model": "meshy-5"
        }
        
        # Submit request
        url = f"{self.base_url}/image-to-3d"
        logger.info(f"Submitting image: {image_path}")
        
        try:
            try:
                response = requests.post(url, headers=self.headers, json=payload)
                # response.raise_for_status()
            except Exception as e:
                print(f"Error submitting image: {e}")
                print(f"Payload: {payload}")
                print(f"URL: {url}")
                print(f"Headers: {self.headers}")
                print(f"Image path: {image_path}")
                pdb.set_trace()

            
            result = response.json()
            task_id = result.get("result")
            if not task_id:
                raise ValueError("No task ID received from API")
            
            logger.info(f"Task submitted successfully. Task ID: {task_id}")
            return task_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a submitted task
        
        Args:
            task_id: The task ID to check
            
        Returns:
            dict: Task status and results
        """
        url = f"{self.base_url}/image-to-3d/{task_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get task status: {e}")
            pdb.set_trace()
            raise
    
    def wait_for_completion(self, task_id: str, timeout: int = 300, check_interval: int = 10) -> Dict[str, Any]:
        """
        Wait for task completion with timeout
        
        Args:
            task_id: The task ID to monitor
            timeout: Maximum time to wait in seconds
            check_interval: How often to check status in seconds
            
        Returns:
            dict: Final task results
        """
        logger.info(f"Waiting for task completion: {task_id}")
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task timed out after {timeout} seconds")
            
            status_data = self.get_task_status(task_id)
            status = status_data.get("status")
            progress = status_data.get("progress", 0)
            
            logger.info(f"Status: {status}, Progress: {progress}%")
            
            if status == "SUCCEEDED":
                logger.info("Task completed successfully!")
                return status_data
            elif status == "FAILED":
                error_msg = status_data.get("task_error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Task failed: {error_msg}")
            elif status in ["IN_PROGRESS", "PENDING"]:
                time.sleep(check_interval)
            else:
                raise RuntimeError(f"Unknown task status: {status}")
    
    def download_model(self, model_url: str, output_path: str) -> None:
        """
        Download a model file from URL
        
        Args:
            model_url: URL to download from
            output_path: Local path to save the file
        """
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded model to: {output_path}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def resize_mesh(self, input_mesh_path: str, output_mesh_path: str, estimated_max_dimension: float) -> None:
        """
        Resize a mesh to a given estimated max dimension
        """
        # load mesh
        mesh = trimesh.load(input_mesh_path)
        # get the largest dimension
        largest_dimension = np.max(mesh.bounds[1] - mesh.bounds[0])
        # resize the mesh
        resized_mesh = mesh.copy()
        resized_mesh.vertices *= estimated_max_dimension / largest_dimension
        # save the mesh
        # pdb.set_trace()
        resized_mesh.export(output_mesh_path)
        return output_mesh_path
    
    def process_image(self, 
                     image_paths: str,
                     output_dir: str,
                     enable_pbr: bool = True,
                     should_remesh: bool = True,
                     should_texture: bool = True,
                     object_type: str = "grasp",
                     download_formats: Optional[list] = None) -> Dict[str, str]:
        """
        Complete pipeline: submit image, wait for completion, download results
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            enable_pbr: Enable PBR materials
            should_remesh: Enable mesh remeshing
            should_texture: Enable texture generation
            download_formats: List of formats to download (e.g., ['obj', 'glb', 'fbx'])
            
        Returns:
            dict: Paths to downloaded files
        """
        if download_formats is None:
            download_formats = ['obj']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # generate complete object
        complete_image_path = self.generate_complete_object(image_paths)
        height = self.query_image_height(complete_image_path)
        # complete_image_path = image_path
        print(f"Complete object generated for {image_paths[0]}, height: {height}")
        # Submit task
        task_id = self.submit_image_to_3d(
            image_path=complete_image_path,
            enable_pbr=enable_pbr,
            should_remesh=should_remesh,
            should_texture=should_texture
        )

        # Wait for completion
        result = self.wait_for_completion(task_id)
        
        # Remesh the object
        # remesh_image_path = self.remesh_object(complete_image_path)
        # print(f"Remesh object generated for {image_path}")

        # Download models
        downloaded_files = {}
        model_urls = result.get("model_urls", {})
        # pdb.set_trace()
        for format_name in download_formats:
            if format_name in model_urls:
                url = model_urls[format_name]
                output_path = os.path.join(output_dir, f"textured_simple_{object_type}.{format_name}")
                tmp_input_path = os.path.join("output", f"textured_simple_{object_type}.{format_name}")
                self.download_model(url, tmp_input_path)
                downloaded_files[format_name] = output_path

        self.resize_mesh(tmp_input_path, output_path, height)
        
        # # Download thumbnail if available
        # thumbnail_url = result.get("thumbnail_url")
        # if thumbnail_url:
        #     thumbnail_path = os.path.join(output_dir, "preview.png")
        #     self.download_model(thumbnail_url, thumbnail_path)
        #     downloaded_files["thumbnail"] = thumbnail_path
        
        # # Download textures if available
        # texture_urls = result.get("texture_urls", [])
        # if texture_urls:
        #     for i, texture_set in enumerate(texture_urls):
        #         for texture_type, url in texture_set.items():
        #             if url:
        #                 texture_path = os.path.join(output_dir, f"textured_map_{texture_type}.png")
        #                 self.download_model(url, texture_path)
        #                 downloaded_files[f"textured_{i}_{texture_type}"] = texture_path
        
        # Save task info
        info_path = os.path.join(output_dir, "task_info.json")
        with open(info_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        downloaded_files["info"] = info_path
        return downloaded_files


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image using Meshy API")
    parser.add_argument("--image_dir", required=True,
                        help="Folder holding the cropped object images "
                             "(<data_root>/<object>/croped_frames)")
    # parser.add_argument("--output_dir", default="output/mesh_1", help="Output directory for results")
    # parser.add_argument("--enable-pbr", action="store_false", default=False, help="Enable PBR materials")
    parser.add_argument("--no-remesh", action="store_true", help="Disable mesh remeshing")
    parser.add_argument("--formats", nargs="+", default=["obj"], 
                       help="Formats to download (obj, glb, fbx, usdz)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    
    args = parser.parse_args()

    image_dir = args.image_dir
    # Get all largest_mask files, excluding those with "complete" in the name
    largest_mask_files = glob.glob(os.path.join(image_dir, "largest_mask*.png"))
    # Filter out files containing "complete" in the name
    largest_mask_files = [f for f in largest_mask_files if "complete" not in f]
    # pdb.set_trace()
    if not largest_mask_files:
        raise FileNotFoundError(f"No largest_mask files found in {image_dir} (excluding complete files)")
    

    grasp_object_files = [f for f in largest_mask_files if "grasp" in f]
    target_object_files = [f for f in largest_mask_files if "target" in f]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(grasp_object_files[0])), "mesh_original")
    os.makedirs(output_dir, exist_ok=True)

    api_key = os.environ.get("MESHY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MESHY_API_KEY is not set. This stage submits the cropped object image to "
            "the Meshy image-to-3D API (https://meshy.ai, paid). Export MESHY_API_KEY "
            "and retry, or drop your own mesh into <object>/mesh/ and skip this stage."
        )
    client = MeshyImageToMesh(api_key)

    # Process grasp object
    results = client.process_image(
        image_paths=grasp_object_files,
        output_dir=output_dir,
        enable_pbr=False, #True,
        should_remesh=not args.no_remesh,
        should_texture=False, #True,
        download_formats=args.formats,
        object_type="grasp"
    )
    if len(target_object_files) > 0:
        results = client.process_image(
            image_paths=target_object_files,
            output_dir=output_dir,
            enable_pbr=False, #True,
            should_remesh=not args.no_remesh,
            should_texture=False, #True,
            download_formats=args.formats,
            object_type="target"
        )
    
    print(f"\n✅ Success! Files saved to: {output_dir}")
    print("Downloaded files:")
    for file_type, file_path in results.items():
        print(f"  {file_type}: {file_path}")
    
    return 0



if __name__ == "__main__":

    main()

