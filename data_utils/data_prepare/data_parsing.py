#进行数据集dairV2X格式转换
import json
import os,imageio
import numpy as np
import shutil
import pyquaternion
import cv2
import copy
import argparse
from tqdm import tqdm

def orthogonalize(matrix):
    u, _, vh = np.linalg.svd(matrix[:3, :3])
    return np.dot(u, vh)

def is_orthogonal(matrix):
    mat = np.array(matrix)
    transpose = mat.T
    product = np.dot(mat, transpose)
    return np.allclose(product, np.eye(mat.shape[0]))

def get_camera_pose(l2c,base_l2w,now_l2w):
    c2l = np.linalg.inv(l2c)
    c2w = now_l2w @ c2l
    c2l_base = np.linalg.inv(base_l2w) @ c2w
    c2l_base[:3,:3] = orthogonalize(c2l_base[:3,:3])
    return c2l_base

def prepare_data(origin_path, dst_path,args):
    base_road = args.base_road
    camera_intrinsics,l2c_extrinsics,l2w_extrinsics,images_path,H_list,W_list = [],[],[],[],[],[]   
    cam_dir = os.path.join(origin_path, args.road_seq)
    cam_list = sorted(os.listdir(cam_dir))
    cam_list = [cam_name for cam_name in cam_list if cam_name != 'coop']
    for cam_name in cam_list:
        print(f"processing {cam_name}")
        cam_path = os.path.join(cam_dir, cam_name)
        calib_path = os.path.join(cam_path, 'calib')
        intrinsic_file = os.path.join(calib_path, 'camera_intrinsic', f'{cam_name}.json')
        extrinsic_file = os.path.join(calib_path, 'lidar2cam', f'{cam_name}.json')
        l2w_extrinsic_file = os.path.join(calib_path, 'lidar2world', f'{cam_name}.json')
        with open(intrinsic_file, 'r') as f:
            intrinsic_data = json.load(f)
        with open(extrinsic_file, 'r') as f:
            extrinsic_data = json.load(f)
        with open(l2w_extrinsic_file, 'r') as f:
            l2w_extrinsic_data = json.load(f)
        intrinsic = np.array(intrinsic_data['intrinsic'])
        extrinsic = np.array(extrinsic_data['lidar2cam'])
        l2w_extrinsic = np.array(l2w_extrinsic_data['lidar2world'])
        
        camera_intrinsics.append(intrinsic)
        l2c_extrinsics.append(extrinsic)
        l2w_extrinsics.append(l2w_extrinsic)
    
        image_path = sorted(os.listdir(os.path.join(cam_path, args.data_type, 'image')))
        image = cv2.imread(os.path.join(cam_path, args.data_type, 'image', image_path[0]))
        H_list.append(image.shape[0])
        W_list.append(image.shape[1])
        images_path.append(image_path)
        
    for i in tqdm(range(len(images_path[0]))):
        if args.sim_num != -1 and i >= args.sim_num:
            break
        camera_extrinsics_tmp = []
        camera_intrinsics_tmp = []
        rgbs = []
        depths = []
        names = []
        for j in range(len(images_path)):
            camera_extrinsic = np.array(get_camera_pose(l2c_extrinsics[j],l2w_extrinsics[base_road],l2w_extrinsics[j]))
            rgb_path = os.path.join(cam_dir,cam_list[j],args.data_type,'image',images_path[j][i])
            depth_path = os.path.join(args.depth_path,cam_list[j],args.data_type,images_path[j][i].replace('.jpg','.png'))
            
            if not os.path.exists(rgb_path):
                print(f"{rgb_path} not found")
                return
                
            if not os.path.exists(depth_path):
                print(f"{depth_path} not found")
                return
            
            camera_intrinsics_tmp.append(camera_intrinsics[j].flatten())
            camera_extrinsics_tmp.append(camera_extrinsic)
            rgbs.append(rgb_path)
            depths.append(depth_path)
            names.append(os.path.splitext(rgb_path.split('/')[-1])[0])
            
        camera_intrinsics_tmp = np.array(camera_intrinsics_tmp)
        camera_extrinsics_tmp = np.array(camera_extrinsics_tmp)
        
        save_path = os.path.join(dst_path, "blender_npz",args.road_seq,args.data_type)
        save_path_list = []
        for cam_name in cam_list:
            save_path_list.append(os.path.join(cam_name,args.data_type))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(
                    os.path.join(save_path, f"{i}.npz"),
                    save_path = save_path_list,
                    names = names,
                    H=H_list,
                    W=W_list,
                    depth=depths,
                    focal_x=camera_intrinsics_tmp[:,0],
                    focal_y=camera_intrinsics_tmp[:,4],
                    u0 = camera_intrinsics_tmp[:,2],
                    v0 = camera_intrinsics_tmp[:,5],
                    rgb=rgbs,
                    extrinsic=camera_extrinsics_tmp #opencv RDF
                )
    print("Done successfully!")
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--home_path', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim', help='path of origin data')
    parse.add_argument('--dataset', type=str, default='standard_rcooper', help='name of dataset')
    parse.add_argument('--road_seq', type=str, default='136-137-138-139', help='name of road sequence')
    parse.add_argument('--base_road',type=int,default=0,help='base road')
    parse.add_argument('--data_type', type=str, default='train', help='data type')
    parse.add_argument('--sim_num', type=int, default=-1, help='sim num')
    args = parse.parse_args()
    args.depth_path = f"{args.home_path}/result/{args.dataset}/depth/{args.road_seq}"
    origin_path = f"{args.home_path}/data/{args.dataset}"
    dst_path = f"{args.home_path}/result/{args.dataset}"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    prepare_data(origin_path, dst_path,args)
    
    