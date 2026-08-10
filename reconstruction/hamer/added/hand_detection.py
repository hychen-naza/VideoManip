from math import nan
from pathlib import Path
import torch
import argparse
import logging
import os
import cv2
import numpy as np
import trimesh
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    iou = interArea / unionArea
    return iou

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()
    # mkdir out_folder if not exists
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    K_path = os.path.join(os.path.dirname(args.img_folder), "cam_K.txt")
    K = np.loadtxt(K_path)
    model, model_cfg = load_hamer(args.checkpoint, estimated_focal_length=int(K[0,0]))

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = sorted(img_paths)
    if not img_paths:
        logger.error(f"no images matching {args.file_type} in {args.img_folder}")
        return 1
    logger.info(f"processing {len(img_paths)} frames from {args.img_folder}")

    # Per-frame outcome counters, reported at the end so a partially-successful
    # run is obvious instead of silently producing a sparse output folder.
    n_ok = 0
    n_skipped_no_depth = 0
    n_skipped_no_hand = 0
    n_skipped_no_hand_depth = 0

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        # get the depth image in the same folder
        depth_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'depth', f'{img_fn}.png')
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            logger.error(f"missing or unreadable depth map for {img_fn}: {depth_path} "
                         f"(run the 'intrinsics' stage first); skipping frame")
            n_skipped_no_depth += 1
            continue
        depth_img = depth_img.astype(np.float32) / 1000

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            # keyp = left_hand_keyp
            # valid = keyp[:,2] > 0.5
            # if sum(valid) > 3:
            #     bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            #     bboxes.append(bbox)
            #     is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            #print(f"sum {sum(valid)}")
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                # check this box isn't overlaying with other boxes
                add_bbox = True
                for other_bbox in bboxes:
                    if bbox_iou(bbox, other_bbox) > 0.5:
                        add_bbox = False
                        break
                if add_bbox:
                    bboxes.append(bbox)
                    is_right.append(1)
        if len(bboxes) == 0:
            logger.info(f"{img_fn}: no confident right-hand detection; skipping frame")
            n_skipped_no_hand += 1
            continue


        # Iterate through all ViTPose outputs and check if keypoints fall within any bounding box
        bboxes_keypoints = [[] for _ in range(len(bboxes))]
        bboxes_keypoints_depth = [[] for _ in range(len(bboxes))]
        H, W = depth_img.shape

        # if (img_fn == 'frame_000274'):
        #     pdb.set_trace()
        for vitposes_idx_, vitposes in enumerate(vitposes_out):
            right_hand_keyp = vitposes['keypoints'][-21:]
            # Check each keypoint against all bounding boxes
            for kpt_idx in range(right_hand_keyp.shape[0]):
                x, y, conf = right_hand_keyp[kpt_idx]
                if conf > 0.5:  # Only draw confident keypoints
                    x, y = int(x), int(y)
                    # Check if keypoint is within any bounding box
                    # if (vitposes_idx_ == 1 and img_fn == 'frame_000274'):
                    #     pdb.set_trace()
                    for i, bbox in enumerate(bboxes):
                        x1, y1, x2, y2 = bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            xc, yc = min(max(x, 0), W-1), min(max(y, 0), H-1)
                            bboxes_keypoints[i].append([xc, yc])
                            try:
                                bboxes_keypoints_depth[i].append(depth_img[yc, xc])
                            except Exception as e:
                                # Out-of-range index or a corrupt depth map: log and
                                # drop this keypoint instead of dropping into a debugger.
                                logger.warning(f"{img_fn}: could not sample depth at "
                                               f"({xc}, {yc}) of {depth_img.shape}: {e}")
                            break  # Found a matching bounding box, no need to check others
        boxes = np.stack(bboxes)
        bboxes_depth = []
        for keyp_depth in bboxes_keypoints_depth:
            # An empty list means no confident keypoint landed inside that box;
            # np.mean would emit a RuntimeWarning and yield nan, which we filter below.
            bboxes_depth.append(np.mean(keyp_depth) if len(keyp_depth) else np.nan)
        bboxes_depth = np.array(bboxes_depth)

        if np.isnan(bboxes_depth).any():
            logger.warning(f"{img_fn}: no valid depth for at least one hand box "
                           f"(depths={bboxes_depth}); skipping frame")
            n_skipped_no_hand_depth += 1
            continue

        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, bboxes_depth, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        all_verts = []
        all_cam_t = []
        all_right = []
        for batch in dataloader:
            batch = recursive_to(batch, device)
            batch_size = batch['img'].shape[0]
            # pdb.set_trace()
            with torch.no_grad():
                out, pred_mano_params = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)

            scaled_focal_length = K[0,0] 
            pred_cam_t_full, scale_factor = cam_crop_to_full(pred_cam, box_center, box_size, img_size, batch['bboxes_depth'], scaled_focal_length)
            pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()
            scale_factor = scale_factor.detach().cpu().numpy()
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Add all verts and cams to listcam_t
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                
        # Render front view
        if len(all_verts) > 0:
            print(f"args.out_folder {args.out_folder}, img_fn {img_fn}")
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
                img_fn = img_fn,
                out_folder = args.out_folder,
                # img_cv2 = img_cv2,
                scale_factor = scale_factor,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), cam_view[:, :, ::-1])
            
            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            # pdb.set_trace()
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
            #pdb.set_trace()
            # save the out['manotorch_params'].cpu(), out['betas'].cpu() info in npy
            mano_params = out['manotorch_params'].cpu().numpy()
            betas = out['betas'].cpu().numpy()
            # put on a dict and save
            mano_data = {}
            pred_mano_params['transl'] = torch.from_numpy(np.array(all_cam_t)).to(device)
            mano_retarget_hand_pose, mano_retarget_hand_joints = model.convert_mano_to_hand_pose_and_joints(pred_mano_params)
            mano_retarget_hand_pose = mano_retarget_hand_pose.cpu().numpy()
            mano_retarget_hand_joints = mano_retarget_hand_joints.cpu().numpy()
            for i in range(out['manotorch_params'].shape[0]):
                person_id = int(batch['personid'][i])
                mano_data[f'person_{person_id}'] = {
                    'mano_params': mano_params[i], # for differential optimization
                    'betas': betas[i],
                    'cam_t': all_cam_t[i],
                    'bboxes_keypoints': bboxes_keypoints[i],
                    'mano_retarget_hand_pose': mano_retarget_hand_pose[i], # for retargeting
                    'mano_retarget_hand_joints': mano_retarget_hand_joints[i], # for retargeting
                }
            # Save the complete dictionary
            np.save(os.path.join(args.out_folder, f'{img_fn}_mano_data.npy'), mano_data)
            n_ok += 1

    logger.info(
        f"hand reconstruction finished: {n_ok}/{len(img_paths)} frames written to "
        f"{args.out_folder} (skipped: {n_skipped_no_hand} no hand detected, "
        f"{n_skipped_no_hand_depth} no valid hand depth, {n_skipped_no_depth} no depth map)"
    )
    if n_ok == 0:
        logger.error("no frames produced hand output - downstream stages will fail")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
