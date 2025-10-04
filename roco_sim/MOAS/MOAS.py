import yaml
import os
import json
import numpy as np
import sys
sys.path.append(f"{os.path.dirname(__file__)}/../../data_utils/vis")
sys.path.append(f"{os.path.dirname(__file__)}/../../data_utils/data_prepare")
from draw_bbox import get_3dbbox_corners
import copy
import random
import argparse
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from Checker import Checker
from Scorer import Scorer
from tqdm import tqdm

random.seed(0)
color_list = [
    # [0,0,0],# original color
    [0.1, 0.1, 0.1],  # black
    [1.0, 1.0, 1.0],  # white
    [0.6, 0.6, 0.6],  # gray
    [0.2, 0.2, 0.5],  # dark blue
    # [0.5, 0.1, 0.2],  # wine red
    [0.3, 0.2, 0.1],  # dark brown
    [0.3, 0.3, 0.2],  # olive green
    # [0.4, 0.3, 0.2],  # dark green
    [0.75, 0.75, 0.75],  # silver
    [0.2, 0.2, 0.3],  # dark gray
    # [0.4, 0.2, 0.3],   # dark red
    [1.0, 1.0, 0.8]   # cream yellow
]


def diverse_place_point(point):
    x = point["3d_location"]["x"]
    y = point["3d_location"]["y"]
    z = point["3d_location"]["z"]
    new_x = x + random.uniform(-1,1)
    new_y = y + random.uniform(-1,1)
    point["3d_location"]["x"] = new_x
    point["3d_location"]["y"] = new_y
    point["3d_location"]["z"] = z
    return point
                

def choose_insert_pos(args,label_temp,car_info,camera_info,no_pos,scorer,pp_checker):
    idx,count,total_count_num = 0,0,50
    for count_num in range(total_count_num):
        count += 1
        grid_idx = scorer.grid_order[idx%len(scorer.grid_order)]
        w_idx = grid_idx//scorer.H_num
        h_idx = grid_idx%scorer.H_num
        possible_pos = scorer.grid_points[w_idx][h_idx]
        if len(possible_pos) == 0:
            idx += 1
            continue
        cam_id_list = np.argsort(scorer.grid_view_score[w_idx][h_idx])[::-1]
        for cam_id in cam_id_list:
            possible_pos_view = scorer.grid_view_points[w_idx][h_idx][cam_id]
            if len(possible_pos_view) != 0:
                break
            
        pos_idx = random.randint(0,len(possible_pos_view)-1)
        if count_num == total_count_num-1:
            no_pos = True
            break
        if count == int(scorer.points_ratio[w_idx][h_idx]*total_count_num)+1:
            idx += 1
            count = 0
        
        # checker
        # check in 3d space
        if 'diverse' in args.insert_way:
            final_pos = diverse_place_point(possible_pos_view[pos_idx])
        else:
            final_pos = possible_pos_view[pos_idx]
            
        if pp_checker.check_occupancy(car_info,final_pos) == False:
            continue
        # check in 2d space
        is_available,item = pp_checker.check_place_available(car_info,final_pos,label_temp,camera_info,args.base_road,args.insert_range)
        if is_available:
            new_bbox = get_3dbbox_corners(item)
            pp_checker.occupancy_map.append(new_bbox)
            break
        
    obj = copy.deepcopy(final_pos)

    return obj,no_pos
            
def insert_car_yaml(args,obj,car_list,car_flag,car_idx,home_path):
    place = np.array([obj['3d_location']['x'],obj['3d_location']['y'],obj['3d_location']['z']-obj['3d_dimensions']['h']/2,1]).T
    insert_pos = place[:3].tolist()
    insert_rot = [0,0,obj['rotation']]
    if args.color_type == 'random':
        color_R = random.uniform(0,1)
        color_G = random.uniform(0,1)
        color_B = random.uniform(0,1)
    else:
        choose_color = random.randint(0,len(color_list)-1)
        color_R = color_list[choose_color][0]
        color_G = color_list[choose_color][1]
        color_B = color_list[choose_color][2]
    size = [car_list[car_idx]["w"],car_list[car_idx]["l"]]
    config = {
                'new_obj_name': car_list[car_idx]["type"]+'_'+str(int(car_flag[car_idx])),
                'blender_file': f'{home_path}/data/blender_file/car/{car_list[car_idx]["type"]}.blend',
                'insert_pos': insert_pos,
                'insert_rot': insert_rot,
                'model_obj_name': 'Car',
                'size': size,
                'target_color': {
                    'material_key': 'car_paint',
                    'color': [color_R, color_G, color_B, 1] # can change the color later
                }
            }
    return place,config   

def generate_blender_yaml(label_path,camera_intrinsics,l2ws,l2cs,args,idx,scorer,pp_checker):
    road_seq,home_path,base_road = args.road_seq,args.home_path,args.base_road
    output_path = f"{args.home_path}/result/{args.dataset}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    label_coop = json.load(open(label_path))
    label_coop = sorted(label_coop, key=lambda x: x['3d_location']['x'])
    
    setting_type = args.save_name
    if args.color_type == 'random':
        car_file_path = os.path.join(home_path,"data/blender_file/car/car_info_all.json")
    else:
        car_file_path = os.path.join(home_path,"data/blender_file/car/car_info.json")
    with open(car_file_path, 'r') as f:
        car_lists = json.load(f)
        
    yaml_dir = os.path.join(output_path, f"standard_sim_{setting_type}",'yaml',f'{road_seq}/{args.data_type}')
    if not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir)
        
    dst_json_path = os.path.join(output_path,f"standard_sim_{setting_type}",f'{road_seq}',f'coop/{args.data_type}/label/lidar_label')
    if not os.path.exists(dst_json_path):
        os.makedirs(dst_json_path)
        
    # place cars in the scene
    enlarge_scale = args.enlarge_scale # simulation amount
    
    pp_checker.init_occupancy_map(label_coop)
    for i in range(enlarge_scale):
        ############################### init grid ##############################
        label_name = os.path.splitext(label_path.split('/')[-1])[0]
        
        if os.path.exists(os.path.join(yaml_dir, label_name+'-'+str(i) + '.yaml')):
            print(f"yaml file {label_name+'-'+str(i)} already exists")
            continue
        if args.draw_map:
            scorer.init_fig()
        scorer.reset_map_scorer()
        scorer.init_grid_occupancy(label_coop)
        pp_checker.occupancy_map = copy.deepcopy(pp_checker.occupancy_map_init)
        label_temp = copy.deepcopy(label_coop)
        
        config = {
            'render_name': label_name+'-'+str(i),
            'output_dir': os.path.join(output_path),
            'road_seq': road_seq,
            'setting_type': f"standard_sim_{setting_type}",
            'scene_file': os.path.join(output_path,"blender_npz",road_seq,args.data_type,str(idx) + '.npz'),
            'hdri_file': args.hdri_file,
        }
        yaml_file_path = os.path.join(yaml_dir, label_name+'-'+str(i) + '.yaml')
        label_file_path = os.path.join(dst_json_path, label_name+'-'+str(i) + '.json')
        
        with open(yaml_file_path, 'w') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        configs = {"mask": []}
            
        with open(yaml_file_path, 'a') as f:
            yaml.dump(configs, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        total_insert_car_num = args.insert_car_num
            
        car_list = copy.deepcopy(car_lists)
        car_flag = np.zeros(len(car_list))
        configs = {"cars": []}
        label_save = copy.deepcopy(label_temp)
       
        no_pos = False 
        region_list = []
        
        ################################### insert car ######################################
        for j in range(total_insert_car_num):
            
            ############################## choose car and insert position ##############################
            car_idx = random.randint(0,len(car_list)-1)
            camera_info = [camera_intrinsics,l2cs,l2ws]
            scorer.score(j,os.path.join(output_path,f"standard_sim_{setting_type}",'grid_map_init',f'{road_seq}',args.data_type,label_name+'-'+str(i)+str(j) + '.png'))
            obj,no_pos = choose_insert_pos(args,label_temp,car_list[car_idx],camera_info,no_pos,scorer,pp_checker)
            if no_pos:
                print(f"======= no pos to insert car =======")
                break    
            ############################ prepare yaml ################################
            place,config = insert_car_yaml(args,obj,car_list,car_flag,car_idx,home_path)
            car_flag[car_idx] += 1
            configs['cars'].append(config)
            
            ############################# prepare label ################################
            obj["type"] = "car"
            obj['3d_location']['z'] = place[2] + car_list[car_idx]["h"]/2
            obj["3d_dimensions"] = {"h":car_list[car_idx]["h"],"w":car_list[car_idx]["w"],"l":car_list[car_idx]["l"]}
            obj["sim"] = 1 # mark this is a simulated car
            label_save.append(obj)
            label_temp.append(obj) 
            scorer.update_grid_occupancy(obj)
                
        if no_pos and len(configs['cars']) < 1:
            os.remove(yaml_file_path)
            continue
        print(f'simulation {idx}, insert num:{len(configs["cars"])},save path:{yaml_file_path}')
        with open(yaml_file_path, 'a') as f:
            yaml.dump(configs, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        with open(label_file_path, 'w') as f:
            json.dump(label_save, f)
        if args.draw_map:
            scorer.save_grid_map(os.path.join(output_path,setting_type,'grid_map',f'{road_seq}',args.data_type,label_name+'-'+str(i) + '.png'))


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="standard_rcooper", help='the path of origin data')
    parser.add_argument('--home_path', type=str, default="/home/ubuntu/duyuwen/RoCo-Sim", help='the path of home directory')
    parser.add_argument('--road_seq', type=str, default='136-137-138-139', help='name of road sequence')
    parser.add_argument('--insert_way',type=str,default='None',help='way to insert car')
    parser.add_argument('--base_road',type=int,default=0,help='base road')
    parser.add_argument('--hdri_file',type=str,default='042.exr',help='hdri file')
    parser.add_argument('--insert_car_num',type=int,default=8,help='insert car num')
    parser.add_argument('--insert_range',type=str,default='0,1',help='insert range')
    parser.add_argument('--enlarge_scale',type=int,default=3,help='enlarge scale')
    parser.add_argument('--color_type',type=str,default='random',help='color type') 
    parser.add_argument('--draw_map',type=bool,default=False,help='draw map')
    parser.add_argument('--view_weight',type=float,default=4,help='view weight')
    parser.add_argument('--occupancy_weight',type=float,default=3,help='occupancy weight')
    parser.add_argument('--cam_num',type=int,default=4,help='camera number')
    parser.add_argument('--sim_num',type=int,default=500,help='simulation number')
    parser.add_argument('--data_type',type=str,default='train',help='data type')
    parser.add_argument('--save_name',type=str,default='',help="save_name")
    args = parser.parse_args()
    args.origin_path = os.path.join(args.home_path,'data',args.dataset)
    args.insert_pos_file = os.path.join(args.home_path,"result",args.dataset,'info',args.road_seq,f'possible_{args.data_type}_car_position.json')
    args.hdri_file = os.path.join(os.path.dirname(args.origin_path),"light",args.hdri_file)
    return args

def load_info(args,cam_list):
    base_road = args.base_road 
    ################################ load camera info ################################
    camera_intrinsics,l2ws,l2cs,H_list,W_list = [],[],[],[],[]
    for cam_name in sorted(cam_list):
        intrinsic_file = os.path.join(args.origin_path, args.road_seq, cam_name, 'calib/camera_intrinsic',f"{cam_name}.json")
        l2c_file = os.path.join(args.origin_path, args.road_seq, cam_name, 'calib/lidar2cam',f"{cam_name}.json")
        l2w_file = os.path.join(args.origin_path, args.road_seq, cam_name, 'calib/lidar2world',f"{cam_name}.json")
        with open(intrinsic_file, 'r') as f:
            intrinsic_data = json.load(f)
        with open(l2c_file, 'r') as f:
            l2c_data = json.load(f)
        with open(l2w_file, 'r') as f:
            l2w_data = json.load(f)
        camera_intrinsics.append(intrinsic_data['intrinsic'])
        l2cs.append(l2c_data['lidar2cam'])
        l2ws.append(l2w_data['lidar2world'])
        image_path = os.path.join(args.origin_path, args.road_seq, cam_name, f'{args.data_type}/image')
        image_list = sorted(os.listdir(image_path))
        H,W = cv2.imread(os.path.join(image_path,image_list[0])).shape[:2]
        H_list.append(H)
        W_list.append(W)
    camera_intrinsics = np.array(camera_intrinsics)
    l2ws = np.array(l2ws)
    l2cs = np.array(l2cs)
    H_list = np.array(H_list)
    W_list = np.array(W_list)
    return camera_intrinsics,l2ws,l2cs,H_list,W_list


if __name__ == '__main__':
    args = parser_args()
    road_label_path = os.path.join(args.origin_path,args.road_seq,'coop',args.data_type,'label','lidar_label')
    cam_list = sorted(os.listdir(args.origin_path + '/' + args.road_seq))
    cam_list = [road for road in cam_list if road != 'coop']
    camera_intrinsics,l2ws,l2cs,args.H,args.W = load_info(args,cam_list)
    args.insert_range = [float(item) for item in args.insert_range.split(',')]
    scorer = Scorer(args)
    scorer.init_map_scorer(args,l2cs=l2cs,l2ws=l2ws,camera_intrinsics=camera_intrinsics)
    pp_checker = Checker(args.H,args.W)
    setting_type = args.save_name
    print(f"============== setting_type: {setting_type} ,data type:{args.data_type}==================")
    dst_path = os.path.join(args.home_path,"result",args.dataset,f"standard_sim_{setting_type}",'yaml',f'{args.road_seq}')
    
    print(f"============== sim_num: {args.sim_num} ==================")
    road_label_files = sorted(os.listdir(road_label_path))
    road_label_files = road_label_files[:args.sim_num]
    for idx, label_file in tqdm(enumerate(road_label_files)):
        generate_blender_yaml(os.path.join(road_label_path,label_file),camera_intrinsics,l2ws,l2cs,args,idx,scorer,pp_checker)