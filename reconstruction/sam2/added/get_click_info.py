import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def get_click_info(video_dir, save_json_path=None, object_name=None):
    """
    Get click information from user interaction on the first frame.
    
    Args:
        first_frame_path (str): Path to the first frame image file
        video_dir (str, optional): Path to the video directory. Used to determine object name if object_name is not provided.
        save_json_path (str, optional): Path to save the clicked information as JSON. If None, saves to object_click.json in the same directory as this file.
        object_name (str, optional): Name of the object/video. If None, will be extracted from video_dir.
    
    Returns:
        list: List of dictionaries, each containing:
            - "clicked_points": List of [x, y] coordinates
            - "clicked_labels": List of labels (1 for positive, 0 for negative)
            - "object_type": "grasp_object" for first object, "target_object" for subsequent objects
    
    Usage:
        - Left click: Add positive point (object)
        - Right click: Add negative point (background)
        - Middle click: Finish current object and start next object
        - Middle click (with no points): Finish all clicking
        - Press 'q' to close window and continue after finishing all objects
    """
    # Local variables for click handling (not global to avoid conflicts)
    current_object_points = []
    current_object_labels = []
    all_objects_data = []
    current_object_idx = 1
    clicking_finished = False
    
    def on_click(event):
        nonlocal current_object_points, current_object_labels, all_objects_data, current_object_idx, clicking_finished
        
        if event.inaxes is not None:
            if event.button == 1:  # Left click - positive point
                current_object_points.append([event.xdata, event.ydata])
                current_object_labels.append(1)
                print(f"Object {current_object_idx}: Added positive point at ({event.xdata:.1f}, {event.ydata:.1f})")
            elif event.button == 3:  # Right click - negative point
                current_object_points.append([event.xdata, event.ydata])
                current_object_labels.append(0)
                print(f"Object {current_object_idx}: Added negative point at ({event.xdata:.1f}, {event.ydata:.1f})")
            elif event.button == 2:  # Middle click - finish current object, start next
                if len(current_object_points) > 0:
                    # Save current object
                    if current_object_idx == 1:
                        all_objects_data.append({
                            "clicked_points": current_object_points.copy(),
                            "clicked_labels": current_object_labels.copy(),
                            "object_type": "grasp_object"
                        })
                    else:
                        all_objects_data.append({
                            "clicked_points": current_object_points.copy(),
                            "clicked_labels": current_object_labels.copy(),
                            "object_type": "target_object"
                        })
                    print(f"Finished object {current_object_idx} with {len(current_object_points)} points. Starting object {current_object_idx + 1}...")
                    # Reset for next object
                    current_object_points = []
                    current_object_labels = []
                    current_object_idx += 1
                else:
                    # No points for current object, finish all clicking
                    clicking_finished = True
                    print("Finished all clicking. Press q to close the window and continue.")
    

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    
    # Sort frames naturally (this handles frame_000001, frame_000002, etc. correctly)
    frame_names.sort()
    #pdb.set_trace()
    if not frame_names:
        print(f"No image files found in {video_dir}")
        return
    
    print(f"Found {len(frame_names)} frames in {video_dir}")
    print(f"First frame: {frame_names[0]}")
    print(f"Last frame: {frame_names[-1]}")

    # Show first frame for user interaction
    first_frame_path = os.path.join(video_dir, frame_names[0])
    first_frame = Image.open(first_frame_path)
    # Load and display the first frame
    
    # Create figure for user interaction
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(first_frame)
    ax.set_title("Click on objects to segment:\nLeft click = positive point, Right click = negative point, Middle click = finish current object & start next\nPress q after finishing all objects")
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Wait for user interaction
    print("Please click on objects in the first frame:")
    print("- Left click: Add positive point (object)")
    print("- Right click: Add negative point (background)")
    print("- Middle click: Finish current object and start next object")
    print("- Middle click (with no points): Finish all clicking")
    print("Press q to close window and continue after finishing all objects")
    
    plt.show()
    # Wait for user to finish clicking
    while not clicking_finished:
        plt.pause(0.1)
    plt.close()
    
    # Save to JSON file if requested
    if save_json_path is not None or object_name is not None:
        # Determine object name
        if object_name is None and video_dir is not None:
            object_name = os.path.basename(os.path.dirname(video_dir))
        elif object_name is None:
            object_name = os.path.splitext(os.path.basename(first_frame_path))[0]
        
        # Determine JSON path
        if save_json_path is None:
            save_json_path = os.path.join(os.path.dirname(__file__), "object_click.json")
        
        # Load existing data or create new dict
        if os.path.exists(save_json_path):
            with open(save_json_path, "r") as f:
                object_click_data = json.load(f)
        else:
            object_click_data = {}
        
        # Save the clicked data
        object_click_data[object_name] = all_objects_data
        with open(save_json_path, "w") as f:
            json.dump(object_click_data, f, indent=2)
        print(f"Saved {len(all_objects_data)} objects' clicked points and labels to {save_json_path}")
    
    return all_objects_data

def main():
    parser = argparse.ArgumentParser(description='Process video frames with SAM2 segmentation')
    parser.add_argument('video_dir', help='Path to folder containing video frames')
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        print(f"Error: Directory {args.video_dir} does not exist")
        return
    print(f"Processing video folder: {args.video_dir}")
    # get object name from the video folder name
    video_name = os.path.basename(os.path.dirname(args.video_dir))
    print(f"Video name: {video_name}")
    # read the object_click.json file
    object_click_path = os.path.join(os.path.dirname(__file__), "object_click.json")
    with open(object_click_path, "r") as f:
        object_click_data = json.load(f)
    if video_name in object_click_data:
        obj_clicked_info = object_click_data[video_name]
    else:
        get_click_info(args.video_dir, save_json_path=object_click_path, object_name=video_name)
        print(f"No click data found for object: {video_name}")


if __name__ == "__main__":
    main()