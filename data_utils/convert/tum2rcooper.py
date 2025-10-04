import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse,math

def get_obj_label(label_file):
    obj_list = []
    label_data = json.load(open(label_file, 'r'))
    box_data = label_data["openlabel"]["frames"]
    for _,frame_obj in box_data.items():
        timestamp = frame_obj["frame_properties"]["timestamp"]
        for _,box in frame_obj["objects"].items():
            for _,obj in box.items():
                info = obj["cuboid"]["val"]
                rotation_quaternion = info[3:7]
                roll, pitch, yaw = R.from_quat(rotation_quaternion).as_euler('xyz')
                obj_json = {
                        "type": obj['type'].lower(),
                        "name": obj['name'].lower(),
                        "timestamp": timestamp,
                        "crowding": 0,
                        "ignore": 0,
                        "occluded_state": 0,
                        "truncated_state": 0,
                        "3d_dimensions": {
                        "h": info[9],
                        "w": info[8],
                        "l": info[7]
                        },
                        "3d_location": {
                        "x": info[0],
                        "y": info[1],
                        "z": info[2]
                        },
                        "rotation": yaw,
                        "track_id": 0
                    }
                obj_list.append(obj_json)
    return obj_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TUM dataset to RCooper dataset')
    parser.add_argument('--data_path', type=str, default='../../data/tumtraf_v2x', help='path to the dataset directory')
    parser.add_argument('--save_path', type=str, default='../../data/rcooper_tum', help='path to save the converted dataset')
    parser.add_argument('--seq_len', type=int, default=40, help='sequence length')
    args = parser.parse_args()
    
    seq_len = args.seq_len
    data_name = ['train','val','test']
    
    cam_name_list = ['north','south1','south2']
    calib_path = os.path.join(args.data_path, 'calib')
    dst_image_dir = os.path.join(args.save_path, 'data','intersection')
    dst_label_dir = os.path.join(args.save_path, 'label','intersection') 
    dst_label_coop_dir = os.path.join(dst_label_dir, 'coop')
    dst_label_dir_ego = [os.path.join(dst_label_dir,cam_name_list[i]) for i in range(len(cam_name_list))]
    
    split_val_file = os.path.join(args.save_path, 'split_data.json')
    val_list = [20,21]
    test_list = [22,23]
    seq_id = 0
    for name in data_name:
        print(f'=================== Processing {name} data ===================')
        
        data_path = os.path.join(args.data_path, name)
        image_path = os.path.join(data_path, 'images')
        lidar_path = os.path.join(data_path, 'point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        label_path = os.path.join(data_path, 'labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        label_list = sorted(os.listdir(label_path))
        lidar_list = sorted(os.listdir(lidar_path))
        cam_id = 0
        for cam_name in os.listdir(image_path):
            if cam_name == 'vehicle_camera_basler_16mm' or 'east' in cam_name:
                continue
            print(f'============== Processing {cam_name.split("_")[-2]} camera ==============')
            image_dir = os.path.join(image_path, cam_name)
            image_list = sorted(os.listdir(image_dir))
            dst_image_path = os.path.join(dst_image_dir, cam_name.split('_')[-2])
            
            for i in range(int(len(image_list)/seq_len)):
                # if cam_id == 0 and name == 'val':
                #     val_list.append(seq_id+i)
                # if cam_id == 0 and name == 'test':
                #     test_list.append(seq_id+i)
                
                image_seq = image_list[i*seq_len:(i+1)*seq_len]
                lidar_seq = lidar_list[i*seq_len:(i+1)*seq_len]
                label_seq = label_list[i*seq_len:(i+1)*seq_len]
                print(f'========== seq {i+seq_id} ==========')
                dst_image_path2 = os.path.join(dst_image_path, f'seq-{i+seq_id}/cam-0')
                dst_label_path = os.path.join(dst_label_coop_dir, f'seq-{i+seq_id}')
                dst_lidar_path = os.path.join(dst_image_path, f'seq-{i+seq_id}/lidar')
                
                if not os.path.exists(dst_image_path2):
                    os.makedirs(dst_image_path2, exist_ok=True)
                if not os.path.exists(dst_label_path):
                    os.makedirs(dst_label_path, exist_ok=True)
                if not os.path.exists(dst_lidar_path):
                    os.makedirs(dst_lidar_path, exist_ok=True)
                    
                for j in range(seq_len):
                    image_file = os.path.join(image_dir, image_seq[j])
                    label_file = os.path.join(label_path, label_seq[j])
                    lidar_file = os.path.join(lidar_path, lidar_seq[j])
                    dst_image_file = os.path.join(dst_image_path2, image_seq[j].split('_')[0]+'.'+image_seq[j].split('_')[1]+'.jpg')
                    os.system(f'cp {image_file} {dst_image_file}')
                    dst_lidar_file = os.path.join(dst_lidar_path, lidar_seq[j].split('_')[0]+'.'+lidar_seq[j].split('_')[1]+'.pcd')
                    os.system(f'cp {lidar_file} {dst_lidar_file}')
                    obj_list = get_obj_label(label_file)
                    dst_label_file = os.path.join(dst_label_path, label_seq[j].split('_')[0]+'.'+label_seq[j].split('_')[1]+'.json')
                    with open(dst_label_file, 'w') as f:
                        json.dump(obj_list, f)
                
            cam_id += 1
        seq_id += int(len(image_list)/seq_len)  
        
    with open(split_val_file, 'w') as f:
        data = {
            "intersection": 
            {
                "val": val_list,
                "test": test_list
            }
        }
        json.dump(data, f)
    
    for label_path in dst_label_dir_ego:
        print(f'Copying labels from {dst_label_coop_dir} to {label_path}')
        if not os.path.exists(label_path):
            os.makedirs(label_path, exist_ok=True)
        os.system(f'cp -r {dst_label_coop_dir}/* {label_path}/')
    