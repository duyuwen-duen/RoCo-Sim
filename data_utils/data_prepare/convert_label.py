import os 
import json,copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_path', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim', help='path to workspace directory')
    parser.add_argument('--dataset', type=str, default='rcooper', help='dataset name')
    parser.add_argument('--sim_data_name', type=str, default='0.0_0.6_16_read-pos-average', help='simulation data name')
    parser.add_argument('--road_seq', type=str, default='117-118-120-119', help='road sequence')
    parser.add_argument('--base_road', type=int, default=0, help='standard road name')
    parser.add_argument('--data_type', type=str, default='train', help='data type')
    args = parser.parse_args()
    origin_data_path = os.path.join(args.home_path, 'data', args.dataset, args.road_seq)
    data_path = os.path.join(args.home_path, 'result', args.dataset, args.sim_data_name, args.road_seq)
    coop_label_path = os.path.join(data_path,'coop',args.data_type,'label/lidar_label')
    cam_list = os.listdir(data_path)
    cam_list = sorted([cam for cam in cam_list if cam != 'coop'])
    l2w_stand_path = os.path.join(origin_data_path, cam_list[args.base_road], 'calib', f'lidar2world/{cam_list[args.base_road]}.json')
    with open(l2w_stand_path, 'r') as f:
        l2w_stand_data = json.load(f)
    l2w_stand = np.array(l2w_stand_data['lidar2world'])
    print(f"================== convert label ==================")
    for cam_name in cam_list:
        os.system(f"cp -r {origin_data_path}/{cam_name}/calib {data_path}/{cam_name}")
        dst_path = f"{data_path}/{cam_name}/{args.data_type}/label/lidar_label"
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        image_dir = os.path.join(data_path, cam_name,args.data_type,"image")
        os.makedirs(dst_path, exist_ok=True)
        calib_path = os.path.join(data_path, cam_name, 'calib')
        
        intrsinic_file = os.path.join(calib_path, f'camera_intrinsic/{cam_name}.json')
        l2c_file = os.path.join(calib_path, f"lidar2cam/{cam_name}.json")
        l2w_file = os.path.join(calib_path, f"lidar2world/{cam_name}.json")
        
        with open(l2w_file, 'r') as f:
            l2w_data = json.load(f)
        l2w = np.array(l2w_data['lidar2world'])
        
        with open(os.path.join(l2c_file,), 'r') as f:
            l2c_data = json.load(f)
        l2c = np.array(l2c_data['lidar2cam'])
        
        with open(intrsinic_file, 'r') as f:
            camera_intrinsic_data = json.load(f)    
        camera_intrinsic = np.array(camera_intrinsic_data['intrinsic'])
        print(f"================== convert label for {cam_name} ==================")
        for image_name in tqdm(sorted(os.listdir(image_dir))):
            image_id,ext = os.path.splitext(image_name)
            origin_label_path = os.path.join(coop_label_path,image_name.replace(ext,'.json'))
            with open(origin_label_path, 'r') as f:
                labels = json.load(f)
            save_labels = copy.deepcopy(labels)
            for label in save_labels:
                loc = np.array([label['3d_location']['x'], label['3d_location']['y'], label['3d_location']['z'],1])
                loc_convert = np.linalg.inv(l2w) @ l2w_stand @ loc
                label['origin_x'] = loc[0]
                label['3d_location']['x'] = loc_convert[0]
                label['3d_location']['y'] = loc_convert[1]
                label['3d_location']['z'] = loc_convert[2]
                theta = label['rotation']
                R_lidar = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                R_world = np.linalg.inv(l2w[:3,:3]) @ l2w_stand[:3,:3] @ R_lidar
                euler_angles = R.from_matrix(R_world).as_euler('xyz', degrees=False)
                label['rotation'] = euler_angles[2]
            
            with open(os.path.join(dst_path,image_name.replace(ext,'.json')), 'w') as f:
                json.dump(save_labels, f)
