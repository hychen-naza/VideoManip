import os
import pdb
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
from pathlib import Path
import re
import json

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Global variables for click handling
current_object_points = []
current_object_labels = []
all_objects_data = []  # List of dicts, each with clicked_points, clicked_labels, object_type
current_object_idx = 1
clicking_finished = False

def on_click(event):
    global current_object_points, current_object_labels, all_objects_data, current_object_idx, clicking_finished
    
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
                        "object_type": f"grasp_object"
                    })
                else:
                    all_objects_data.append({
                        "clicked_points": current_object_points.copy(),
                        "clicked_labels": current_object_labels.copy(),
                        "object_type": f"target_object"
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

def process_video_folder(video_dir, output_dir=None, clicked_info=None, object_mask=True):
    # Set output directory
    if output_dir is None:
        output_dir = video_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
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
    print(f"Saving masks to: {output_dir}")
    
    # Initialize SAM2 predictor
    #pdb.set_trace()
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    # Show first frame for user interaction
    first_frame_path = os.path.join(video_dir, frame_names[0])
    first_frame = Image.open(first_frame_path)
    
    if clicked_info is None:
        # Reset global variables for new clicking session
        global current_object_points, current_object_labels, all_objects_data, current_object_idx, clicking_finished
        current_object_points = []
        current_object_labels = []
        all_objects_data = []
        current_object_idx = 1
        clicking_finished = False
        
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

        # Save all objects' clicked points and labels to json file
        object_click_path = os.path.join(os.path.dirname(__file__), "object_click.json")
        with open(object_click_path, "r") as f:
            object_click_data = json.load(f)
        object_name = os.path.basename(os.path.dirname(video_dir))
        # Save as a list of dicts
        object_click_data[object_name] = all_objects_data
        with open(object_click_path, "w") as f:
            json.dump(object_click_data, f)
        print(f"Saved {len(all_objects_data)} objects' clicked points and labels to {object_click_path}")
        clicked_info = all_objects_data

    # Ensure clicked_info is a list
    if clicked_info is None:
        print("Error: No clicked info provided and no user interaction occurred.")
        return
    
    if not isinstance(clicked_info, list):
        print(f"Error: clicked_info should be a list, got {type(clicked_info)}")
        return
    
    if len(clicked_info) == 0:
        print("Error: clicked_info is empty.")
        return

    ann_obj_id = 0
    for frame_idx, click_dict in enumerate(clicked_info):
        # Convert clicked points to numpy arrays
        points = np.array(click_dict["clicked_points"], dtype=np.float32)
        labels = np.array(click_dict["clicked_labels"], dtype=np.int32)
        object_type = click_dict["object_type"]
        # Process the first frame with selected points
        ann_frame_idx = 0
        ann_obj_id += 1
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        
        # Run propagation throughout the video
        video_segments = {}
        print("Propagating segmentation through all frames...")
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Save masks for all frames
        print("Saving masks...")
        for out_frame_idx in range(len(frame_names)):
            if out_frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    # Save as PNG with original frame filename
                    mask_uint8 = (out_mask * 255).astype(np.uint8)
                    if len(mask_uint8.shape) > 2:
                        mask_uint8 = mask_uint8.squeeze()
                    
                    # Use original frame filename for the mask
                    original_frame_name = os.path.splitext(frame_names[out_frame_idx])[0]
                    mask_filename = f"{original_frame_name}_{object_type}.png"
                    mask_path = os.path.join(output_dir, mask_filename)
                    cv2.imwrite(mask_path, mask_uint8)
             
                    # Save as numpy array
                    # npy_filename = f"{out_frame_idx}.npy"
                    # npy_path = os.path.join(output_dir, npy_filename)
                    # np.save(npy_path, mask_uint8)
            # if out_frame_idx == 0 and object_mask:
            #     original_frame_name = frame_names[out_frame_idx]
            #     mask_filename = original_frame_name
            #     output_dir_first_frame = os.path.join(os.path.dirname(output_dir), "masks")
            #     #output_dir.replace("masks_pred", "masks_pred_first_frame")
            #     mask_path = os.path.join(output_dir_first_frame, mask_filename)
            #     cv2.imwrite(mask_path, mask_uint8)
        print(f"Processing complete! Masks saved for {len(video_segments)} frames in {output_dir}.")


        if object_mask:
            # save the cropped frame with largest masked area   
            largest_mask_area = 0
            largest_mask_frame = None
            largest_mask_obj_id = None
            largest_mask_bbox = None
            # save the croped frames that contain the segmented object in another folder 
            croped_frames_dir = os.path.join(os.path.dirname(output_dir), "croped_frames")
            os.makedirs(croped_frames_dir, exist_ok=True)
            for out_frame_idx in range(5): #len(frame_names)
                if out_frame_idx in video_segments and ann_obj_id in video_segments[out_frame_idx]:
                    out_mask = video_segments[out_frame_idx][ann_obj_id]
                    
                    original_frame_name = frame_names[out_frame_idx]
                    original_frame_path = os.path.join(video_dir, original_frame_name)
                    original_frame = Image.open(original_frame_path)

                    # Convert mask to numpy array if not already
                    if not isinstance(out_mask, np.ndarray):
                        mask_np = np.array(out_mask)
                    else:
                        mask_np = out_mask

                    # Ensure mask is 2D
                    if mask_np.ndim > 2:
                        mask_np = mask_np.squeeze()
                    if mask_np.ndim != 2:
                        raise ValueError(f"Mask should be 2D after squeezing, got shape {mask_np.shape}")

                    # Find bounding box of the mask
                    ys, xs = np.where(mask_np)
                    if len(xs) == 0 or len(ys) == 0:
                        continue  # skip empty masks
                    x_min, x_max = max(0, xs.min() - 40), min(original_frame.width-1, xs.max()+ 40)
                    y_min, y_max = max(0, ys.min() - 40), min(original_frame.height-1, ys.max()+ 40)
                    bbox = (x_min, y_min, x_max+1, y_max+1)  # PIL crop is (left, upper, right, lower)

                    # Calculate mask area
                    mask_area = np.sum(mask_np)
                    if mask_area > largest_mask_area:
                        # Save "before" only when this mask becomes the largest (same condition as "after")
                        largest_mask_area = mask_area
                        largest_mask_frame = out_frame_idx
                        largest_mask_obj_id = out_obj_id
                        largest_mask_bbox = bbox

                        # Crop the frame
                        # cropped_frame = original_frame.crop(bbox)
                        # # Apply the mask to the cropped frame with white background
                        # mask_cropped = Image.fromarray(mask_np[y_min:y_max+1, x_min:x_max+1].astype(np.uint8) * 255)
                        # # Create a white background image
                        # white_bg = Image.new('RGB', cropped_frame.size, (255, 255, 255))
                        # # Composite the cropped frame over white background using the mask
                        # # pdb.set_trace()
                        # # print(bbox, original_frame.size, mask_np.shape)
                        # # print(out_frame_idx, cropped_frame.size, white_bg.size, mask_cropped.size)
                        # cropped_frame = Image.composite(cropped_frame, white_bg, mask_cropped)
                        # # Save
                        # cropped_frame_path = os.path.join(croped_frames_dir, f"{os.path.splitext(original_frame_name)[0]}.png")
                        # cropped_frame.save(cropped_frame_path)
                
                       
            # Save the frame with largest masked area
            if largest_mask_frame is not None and largest_mask_bbox is not None:
                largest_frame_name = frame_names[largest_mask_frame]
                largest_frame_path = os.path.join(video_dir, largest_frame_name)
                largest_frame = Image.open(largest_frame_path)
                
                # Crop the largest frame
                largest_cropped = largest_frame.crop(largest_mask_bbox)
                largest_cropped_path = os.path.join(croped_frames_dir, f"largest_mask_{object_type}_{largest_frame_name}")
                largest_cropped.save(largest_cropped_path)

                # Get the mask for the largest area
                largest_mask = video_segments[largest_mask_frame][largest_mask_obj_id]
                if not isinstance(largest_mask, np.ndarray):
                    largest_mask_np = np.array(largest_mask)
                else:
                    largest_mask_np = largest_mask
                
                if largest_mask_np.ndim > 2:
                    largest_mask_np = largest_mask_np.squeeze()
                
                # Apply mask to cropped frame with white background
                x_min, y_min, x_max, y_max = largest_mask_bbox
                # Crop the mask to match the cropped frame
                mask_cropped_np = largest_mask_np[y_min:y_max, x_min:x_max]
                mask_cropped = Image.fromarray(mask_cropped_np.astype(np.uint8) * 255)
                # Create a white background image
                white_bg = Image.new('RGB', largest_cropped.size, (255, 255, 255))
                # Composite the cropped frame over white background using the mask
                largest_cropped = Image.composite(largest_cropped, white_bg, mask_cropped)
                
                # Save the largest cropped frame
                largest_cropped_path = os.path.join(croped_frames_dir, f"largest_mask_bg_white_{object_type}_{largest_frame_name}")
                largest_cropped.save(largest_cropped_path)
                
                print(f"Saved largest masked area frame: {largest_cropped_path}")
                print(f"Frame: {largest_frame_name}, Object ID: {largest_mask_obj_id}, Area: {largest_mask_area} pixels")
            else:
                print("No valid largest mask found to save.")


def main():
    parser = argparse.ArgumentParser(description='Process video frames with SAM2 segmentation')
    parser.add_argument('video_dir', help='Path to folder containing video frames')
    parser.add_argument('--hand_input_folder', help='Path to folder containing hand frames')
    parser.add_argument('--obj_output_dir', help='Path to save segmentation masks (default: same as video_dir)')
    parser.add_argument('--hand_output_dir', help='Path to save segmentation masks (default: same as video_dir)')
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
        #obj_clicked_points = object_click_data[video_name] #["clicked_points"]
        #obj_clicked_labels = object_click_data[video_name] #["clicked_labels"]
    else:
        obj_clicked_info = None
        print(f"No click data found for object: {video_name}")
        # return
    # process the video folder to get object mask
    process_video_folder(args.video_dir, args.obj_output_dir, clicked_info=obj_clicked_info)
    
    ### plot the human hand clicked points on the first frame ###

    # first_frame_path = os.path.join(args.video_dir, "frame_000000.png")
    # first_frame = Image.open(first_frame_path)
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.imshow(first_frame)
    # show_points(np.array(human_hand_clicked_points), np.array(human_hand_clicked_labels), ax)
    # plt.savefig(os.path.join(args.hand_output_dir, "human_hand_clicked_points.png"))
    # pdb.set_trace()
    # process the video folder to get human hand mask
    # import glob
    # hand_datas = sorted(glob.glob(os.path.join(args.hand_input_folder, "*.npy")))
    # # take the keypoints from the first frame
    # hand_data = np.load(hand_datas[0], allow_pickle=True).item()

    # human_hand_clicked_points = []
    # human_hand_clicked_labels = []
    # for key, value in hand_data.items():
    #     if 'bboxes_keypoints' in value:
    #         human_hand_clicked_points.extend(value['bboxes_keypoints'])
    #         human_hand_clicked_labels.extend([1] * len(value['bboxes_keypoints']))
    #     else:
    #         # human_hand_clicked_points = None
    #         # human_hand_clicked_labels = None
    #         print(f"No click data found for human hand")
    # process_video_folder(args.video_dir, args.hand_output_dir, saved_clicked_points=human_hand_clicked_points, saved_clicked_labels=human_hand_clicked_labels, object_mask=False)

if __name__ == "__main__":
    # Example usage - you can modify this or use command line arguments
    # process_video_folder("/path/to/your/frames")
    
    # For command line usage, uncomment the line below and comment out the example above
    main()
