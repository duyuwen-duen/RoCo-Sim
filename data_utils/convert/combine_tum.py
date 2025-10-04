import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse,math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TUM dataset to RCooper dataset')
    parser.add_argument('--data_path', type=str, default='../../data/tumtraf_v2x', help='path to the dataset directory')
    parser.add_argument('--save_path', type=str, default='../../data/tumtraf_v2x_new', help='path to save the converted dataset')
    args = parser.parse_args()
    
    data_name = ['train', 'val']
    for name in data_name:
        print(f'=================== Processing {name} data ===================')
        
        data_path = os.path.join(args.data_path, name)
        image_path = os.path.join(data_path, 'images')
        lidar_path = os.path.join(data_path, 'point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        label_path = os.path.join(data_path, 'labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        dst_lidar_path = os.path.join(args.save_path, 'train', 'point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        dst_label_path = os.path.join(args.save_path, 'train', 'labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        if not os.path.exists(dst_lidar_path):
            os.makedirs(dst_lidar_path)
        if not os.path.exists(dst_label_path):
            os.makedirs(dst_label_path)
        os.system(f'cp -r {lidar_path}/* {dst_lidar_path}')
        os.system(f'cp -r {label_path}/* {dst_label_path}')
        for cam_name in os.listdir(image_path):
            if cam_name == 'vehicle_camera_basler_16mm' or 'east' in cam_name:
                continue
            print(f'============== Processing {cam_name.split("_")[-2]} camera ==============')
            dst_image_path = os.path.join(args.save_path, 'train', 'images', cam_name)
            if not os.path.exists(dst_image_path):
                os.makedirs(dst_image_path)
            os.system(f'cp -r {os.path.join(image_path, cam_name)}/* {dst_image_path}')
            
        
    