from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import argparse
import torch
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def get_points(label,K,l2c):
    input_coords = []
    input_labels = []
    for obj in label:
        loc = np.array([obj['3d_location']['x'], obj['3d_location']['y'], obj['3d_location']['z'],1])
        loc_3d = l2c @ loc.reshape(4,1)
        loc_3d = loc_3d[:3] / loc_3d[3]
        loc_2d = K @ loc_3d
        loc_2d = loc_2d[:2] / loc_2d[2]
        if loc_2d[0] < 0 or loc_2d[0] > 1920 or loc_2d[1] < 0 or loc_2d[1] > 1200:
            continue
        input_coords.append(loc_2d[:2])
        input_labels.append(1)
    input_coords = np.array(input_coords).squeeze()
    input_labels = np.array(input_labels)
    print(input_coords.shape, input_labels.shape)
    return input_coords, input_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="./image/", help="path to image file")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--output_dir", type=str, default="./output", help="image save path")
    parser.add_argument("--calib_path", type=str, default="/home/ubuntu/duyuwen/ChatSim_v2x/data/rcooper/calib/lidar2cam/136.json", help="path to intrinsic file")
    parser.add_argument("--cam_id", type=int, default=0, help="camera id")
    args = parser.parse_args()

    cam_id = args.cam_id
    model_type = "vit_t"
    sam_checkpoint = "./weights/mobile_sam.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)
    label_path = args.img_path.replace("image", "label")
    label_list = os.listdir(label_path)
    label_list.sort()

    with open(args.calib_path, 'r') as f:
        camera_info = json.load(f)
    camera_intrinsic = np.array(camera_info[f'cam_{cam_id}']['intrinsic'])
    l2c = np.array(camera_info[f'cam_{cam_id}']['extrinsic'])

    idx = 0
    for image_name in sorted(os.listdir(args.img_path)):
        image_path = os.path.join(args.img_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        label_file = os.path.join(label_path, label_list[idx])
        with open(label_file, 'r') as f:
            label = json.load(f)
        input_coords, input_labels = get_points(label,camera_intrinsic,l2c)
        masks, _, _ = predictor.predict(point_coords=input_coords, point_labels=input_labels)
        background = np.zeros_like(image)[:,:,0]
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        mask = masks[0]
        background = np.where(mask == 0, background, mask)
        show_mask(mask, plt.gca())
        show_points(input_coords, input_labels, plt.gca())
        cv2.imwrite(os.path.join(args.output_dir, image_name), background*255)
        print(f"Image {image_name} saved to {args.output_dir}")
        plt.savefig(os.path.join(args.output_dir, f"{image_name}.png"))
        idx += 1
