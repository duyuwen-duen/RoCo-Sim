import os
import cv2
import time
import json
import argparse
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from tqdm import tqdm
import sys

sys.path.append(f"{os.path.dirname(__file__)}/../data_utils/vis")
from draw_bbox import get_2dbbox_corners, get_3dbbox_corners


def error_loss(ratio, depth, points_2d, point_clouds, H, W):
    k1, b1, k2, b2 = ratio
    error = 0
    for i in range(points_2d.shape[1]):
        x, y = points_2d[:, i]
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        if point_clouds[2, i] < 0:
            continue

        dis = np.linalg.norm(point_clouds[:, i])
        weight = np.power(dis / 50, 2) if dis > 50 else 1
        k = k1 * depth[int(y), int(x)] + b1
        b = k2 * depth[int(y), int(x)] + b2

        pred_z = k * depth[int(y), int(x)] + b
        error += weight * np.square(pred_z - point_clouds[2, i])
    return error


def filter_points(points_cloud, points_2d, mask, H, W):
    mask = mask.astype(bool)
    filtered_clouds = []
    filtered_2d = []

    for i in range(points_2d.shape[1]):
        x, y = points_2d[:, i]
        if 0 <= int(x) < W and 0 <= int(y) < H and mask[int(y), int(x)]:
            filtered_clouds.append(points_cloud[:, i])
            filtered_2d.append(points_2d[:, i])

    return np.array(filtered_clouds).T, np.array(filtered_2d).T


def run_depth_estimation(args, cam_list, cam_dir):
    print('\033[92m#### DEPTH ESTIMATION ####\033[0m')
    for cam_name in cam_list:
        image_path = os.path.join(cam_dir, cam_name, args.data_type, 'image')
        depth_path = os.path.join(args.depth_dir, cam_name, args.data_type)

        if not os.path.exists(depth_path):
            os.makedirs(depth_path)

        existing = len(os.listdir(depth_path))
        if existing == args.sim_num:
            print(f'[SKIP] {depth_path} already generated.')
            continue

        range_str = f'{existing},{args.sim_num}'
        os.chdir(f'{args.home_path}/roco_sim/background/Depth-Anything-V2/metric_depth/')
        os.system(f'python run.py --img-path {image_path} --outdir {depth_path} --data_range {range_str}')
        os.chdir(args.home_path)


def run_segmentation(args, cam_list, cam_dir, seg_dir):
    print('\033[92m#### SEGMENTATION ####\033[0m')
    for cam_name in cam_list:
        cam_path = os.path.join(cam_dir, cam_name)
        image_path = os.path.join(cam_path, args.data_type, 'image')
        label_path = os.path.join(cam_path, args.data_type, 'label/lidar_label')
        seg_path = os.path.join(seg_dir, cam_name, args.data_type)

        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        existing = len(os.listdir(seg_path))
        if existing == args.sim_num:
            print(f'[SKIP] {seg_path} already generated.')
            continue

        range_str = f'{existing},{args.sim_num}'
        os.chdir(f'{args.home_path}/roco_sim/background/MobileSAM/MobileSAMv2/')
        os.system(f'python Inference.py --img_path {image_path} --label_path {label_path}/ '
                  f'--output_dir {seg_path} --encoder_type efficientvit_l2 --gpu_id {args.gpu_id} '
                  f'--calib_path {cam_path}/calib --cam_name {cam_name} --data_range {range_str}')
        os.chdir(args.home_path)


def run_depth_alignment(args, cam_list, cam_dir, seg_dir):
    print('\033[92m#### DEPTH ALIGNMENT ####\033[0m')
    for cam_name in cam_list:
        print(f'Aligning depth for {cam_name}')
        cam_path = os.path.join(cam_dir, cam_name)
        intrinsic_file = os.path.join(cam_path, 'calib/camera_intrinsic', f'{cam_name}.json')
        extrinsic_file = os.path.join(cam_path, 'calib/lidar2cam', f'{cam_name}.json')

        with open(intrinsic_file, 'r') as f:
            intrinsic = np.array(json.load(f)['intrinsic'])
        with open(extrinsic_file, 'r') as f:
            extrinsic = np.array(json.load(f)['lidar2cam'])

        image_path = os.path.join(cam_path, args.data_type, 'image')
        label_path = os.path.join(cam_path, args.data_type, 'label/lidar_label')
        depth_path = os.path.join(args.depth_dir, cam_name, args.data_type)
        lidar_path = os.path.join(cam_path, args.data_type, 'lidar')
        seg_path = os.path.join(seg_dir, cam_name, args.data_type)
        aligned_path = depth_path.replace('depth_origin', 'depth')

        if not os.path.exists(aligned_path):
            os.makedirs(aligned_path)

        for idx, pcd_file in enumerate(sorted(os.listdir(lidar_path))):
            if args.sim_num != -1 and idx >= args.sim_num:
                break

            pcd = o3d.io.read_point_cloud(os.path.join(lidar_path, pcd_file))
            points = np.asarray(pcd.points)
            points = points[points[:, 0] > 0]

            points_h = np.vstack((points.T, np.ones(points.shape[0])))
            points_cam = (extrinsic @ points_h)[:3]
            points_2d = (intrinsic @ points_cam)[:2] / points_cam[2:3]

            depth_file = os.path.join(depth_path, sorted(os.listdir(depth_path))[idx])
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
            rgb = cv2.imread(os.path.join(image_path, depth_file.split('/')[-1].replace('.png', '.jpg')))

            seg_file = os.path.join(seg_path, depth_file.split('/')[-1].replace('.png', '.jpg'))
            if not os.path.exists(seg_file):
                seg_file = seg_file.replace('.jpg', '.png')
            mask = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)

            f_points_cam, f_points_2d = filter_points(points_cam, points_2d, mask, rgb.shape[0], rgb.shape[1])

            if f_points_2d.size == 0:
                print(f"[SKIP] No valid lidar points for {depth_file}")
                depth_masked = np.where(mask[..., None] == 0, 65535, depth[..., None]).astype(np.uint16)
                cv2.imwrite(os.path.join(aligned_path, depth_file.split('/')[-1]), depth_masked)
                continue

            if np.sum(depth == 655.35) > 1000:
                print(f"[SKIP] Already aligned {depth_file}")
                continue

            result = minimize(error_loss, [1, 0, 1, 0], args=(depth, f_points_2d, f_points_cam, rgb.shape[0], rgb.shape[1]), method='BFGS')
            k1, b1, k2, b2 = result.x

            aligned = (k1 * depth + b1) * depth + (k2 * depth + b2)
            aligned = np.expand_dims(aligned, -1)
            final_depth = np.where(mask[..., None] == 0, 65535, aligned * 100).astype(np.uint16)

            cv2.imwrite(os.path.join(aligned_path, depth_file.split('/')[-1]), final_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_path', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim')
    parser.add_argument('--dataset', type=str, default='standard_rcooper')
    parser.add_argument('--road_seq', type=str, default='136-137-138-139')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--sim_num', type=int, default=-1)
    args = parser.parse_args()

    args.data_path = os.path.join(args.home_path, 'data', args.dataset)
    args.depth_dir = os.path.join(args.home_path, 'result', args.dataset, 'depth_origin', args.road_seq)
    seg_dir = args.depth_dir.replace('depth_origin', 'seg')
    cam_dir = os.path.join(args.data_path, args.road_seq)
    cam_list = sorted([c for c in os.listdir(cam_dir) if c != 'coop'])
    if not os.path.exists(args.depth_dir):
        os.makedirs(args.depth_dir)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    start = time.time()
    run_depth_estimation(args, cam_list, cam_dir)
    run_segmentation(args, cam_list, cam_dir, seg_dir)
    run_depth_alignment(args, cam_list, cam_dir, seg_dir)
    print(f"✅ All finished in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
