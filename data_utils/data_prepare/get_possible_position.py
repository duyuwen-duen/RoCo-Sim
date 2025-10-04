import numpy as np
import json
import argparse
import copy
import sys,os
sys.path.append(f"{os.path.dirname(__file__)}/../vis")
from draw_bbox import project_to_image
import os,cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import random


def check_in_image(label, l2cs, camera_intrinsics, l2ws,base_road,H_list,W_list):
    in_image_num = np.zeros(len(l2cs))
    for idx_i in range(len(l2cs)):
        item = copy.deepcopy(label)
        location_lidar = np.linalg.inv(l2ws[idx_i]) @ l2ws[base_road] @ np.array([item['3d_location']['x'],item['3d_location']['y'],item['3d_location']['z'],1]).T
        point,depth = project_to_image(location_lidar[:3], camera_intrinsics[idx_i], l2cs[idx_i])
        if (point[0]>=0) & (point[0]<W_list[idx_i]) & (point[1]>=0) & (point[1]<H_list[idx_i]) & (depth[0] > 0):
            in_image_num[idx_i] = 1
    return np.array(in_image_num)


def check_distance(label, l2cs, camera_intrinsics, l2ws,base_road,H_list,W_list,max_dis):
    distance = np.zeros(len(l2cs))
    for idx_i in range(len(l2cs)):
        item = copy.deepcopy(label)
        location_lidar = np.linalg.inv(l2ws[idx_i]) @ l2ws[base_road] @ np.array([item['3d_location']['x'],item['3d_location']['y'],item['3d_location']['z'],1]).T
        location_cam = l2cs[idx_i] @ location_lidar
        distance[idx_i] = np.linalg.norm(location_cam[:3])
    if np.min(distance) < max_dis:
        return True
    else:
        return False


def get_camera_info(args,road_list):
    l2cs,l2ws,intrinsics,H_list,W_list = [],[],[],[],[]
    for road_name in sorted(road_list):
        intrinsic_file = os.path.join(args.origin_path, args.road_seq, road_name, 'calib/camera_intrinsic',f"{road_name}.json")
        l2c_file = os.path.join(args.origin_path, args.road_seq, road_name, 'calib/lidar2cam',f"{road_name}.json")
        l2w_file = os.path.join(args.origin_path, args.road_seq, road_name, 'calib/lidar2world',f"{road_name}.json")
        with open(intrinsic_file, 'r') as f:
            intrinsic_data = json.load(f)
        with open(l2c_file, 'r') as f:
            l2c_data = json.load(f)
        with open(l2w_file, 'r') as f:
            l2w_data = json.load(f)
        intrinsics.append(np.array(intrinsic_data['intrinsic']))
        l2cs.append(np.array(l2c_data['lidar2cam']))
        l2ws.append(np.array(l2w_data['lidar2world']))
        image_path = os.path.join(args.origin_path, args.road_seq, road_name, f'{args.data_type}/image')
        image_list = sorted(os.listdir(image_path))
        image = cv2.imread(os.path.join(image_path, image_list[0]))
        H,W,C = image.shape
        H_list.append(H)
        W_list.append(W)
    return l2cs,l2ws,intrinsics,H_list,W_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_path', type=str, default="/home/ubuntu/duyuwen/RoCo-Sim/", help='the path of project')
    parser.add_argument('--dataset', type=str, default='standard_rcooper', help='the name of dataset')
    parser.add_argument('--road_seq', type=str, default='117-118-120-119', help='name of road sequence')
    parser.add_argument('--base_road',type=int,default=0,help='the index of base road')
    parser.add_argument('--max_dis',type=float,default=50,help='the min distance of possible position')
    parser.add_argument('--data_type',type=str,default='train',help='the type of data')
    args = parser.parse_args()
    max_dis = args.max_dis
    args.origin_path = args.home_path + '/data/' + args.dataset
    args.save_path = args.home_path + '/result/' + args.dataset + '/info'
    
    base_road = 0
    road_list = sorted(os.listdir(args.origin_path + '/' + args.road_seq))
    road_list = [road for road in road_list if road != 'coop']
    base_road_name = road_list[base_road]
    l2cs,l2ws,intrinsics,H_list,W_list = get_camera_info(args,road_list)
    
    save_car_path = args.save_path + '/' + args.road_seq +'/'+f'possible_{args.data_type}_car_position.json'
    if not os.path.exists(os.path.dirname(save_car_path)):
        os.makedirs(os.path.dirname(save_car_path))
        
    label_path = args.origin_path + '/'+ args.road_seq + f'/coop/{args.data_type}/label/lidar_label'
    
    replace_car_type = ['car','van','truck','bus']

    car_list = []
    choose_car_num = np.zeros(len(l2cs))
    label_list = sorted(os.listdir(label_path))
    for label_file in label_list:
        with open(os.path.join(label_path, label_file), 'r') as f:
            labels = json.load(f)
        for label in labels:
            in_image_num = check_in_image(label, l2cs, intrinsics, l2ws,base_road,H_list,W_list)
            if max_dis != -1:
                if check_distance(label, l2cs, intrinsics, l2ws,base_road,H_list,W_list,max_dis) == False:
                    continue
            if np.sum(in_image_num) > 0:
                if label['occluded_state'] != 0 and label['occluded_state'] != 1:
                    continue
                for idx_i in range(len(in_image_num)):
                    if in_image_num[idx_i] == 1:
                        choose_car_num[idx_i] += 1
                car_list.append(label)
                point = [label['3d_location']['x'],label['3d_location']['y'],label['3d_location']['z']]
                    
        print(f"possible car position: {choose_car_num},")
                    
    car_list = sorted(car_list, key=lambda x: (x['3d_location']['x']))
    print(f'total possible position car number {len(car_list)},save to {save_car_path}')
    with open(save_car_path, 'w') as f:
        json.dump(car_list, f, indent=4)
    