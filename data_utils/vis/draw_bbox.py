import numpy as np
import cv2,os
import json
from pyquaternion import Quaternion
import argparse
from scipy.spatial.transform import Rotation as R
import math
import re
from tqdm import tqdm

     
def project_to_image(points, K,l2c):
    """
    transform points (x,y,z) to image (h,w,1)
    """
    points = np.array(points).reshape((3, 1))
    R = l2c[:3,:3]
    T = l2c[:3,3]
    points_camera = np.dot(R, points) + T.reshape((3, 1))
    points_2d_homogeneous = np.dot(K, points_camera)
    
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    return points_2d,points_2d_homogeneous[2,:]

def get_3dbbox_corners(label):
    location = [label['3d_location']['x'], label['3d_location']['y'], label['3d_location']['z']]
    size = [label['3d_dimensions']['l'],label['3d_dimensions']['w'],label['3d_dimensions']['h']]
    theta = label['rotation']
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    
    points = np.array([[-size[0]/2,-size[1]/2,-size[2]/2],
                    [size[0]/2,-size[1]/2,-size[2]/2],
                    [size[0]/2,size[1]/2,-size[2]/2],
                    [-size[0]/2,size[1]/2,-size[2]/2],
                    [-size[0]/2,-size[1]/2,size[2]/2],
                    [size[0]/2,-size[1]/2,size[2]/2],
                    [size[0]/2,size[1]/2,size[2]/2],
                    [-size[0]/2,size[1]/2,size[2]/2]])
    points = np.dot(Rz, points.T).T
    
    input_bbox = location + points
    
    return input_bbox


def draw_3dbbox_func(img, input_bbox,l2c,camera_intrinsic,name,label):
    output_bbox = []
    depth_list = []
    for point in input_bbox:
        project_point,depth = project_to_image(point, camera_intrinsic, l2c)
        output_bbox.append(project_point.flatten())
        depth_list.append(depth)
    depth_list = np.array(depth_list)
    output_bbox = np.array(output_bbox)
    points = output_bbox.astype(np.int32)
    # for i in range(0, 8):
    #     cv2.putText(img, str(i), tuple(points[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    is_inimage = (depth_list > 0) & (points[:,0] >= 0) & (points[:,0] < img.shape[1]) & (points[:,1] >= 0) & (points[:,1] < img.shape[0]) 
    if not np.any(is_inimage):
        return
    if 'sim' in label.keys():
        color = (255, 215, 0)
    else:
        color = (0, 255, 0)
    
    for i in range(0, 4):
        cv2.line(img, tuple(points[i]), tuple(points[(i+1)%4]), color, 4)
        cv2.line(img, tuple(points[i+4]), tuple(points[(i+1)%4+4]), color, 4)
        cv2.line(img, tuple(points[i]), tuple(points[i+4]), color, 4)
    # cv2.putText(img, f"{label['3d_location']['x']:.4f},{label['3d_location']['y']:.4f},{label['3d_location']['z']:.4f}", tuple(points[7]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
def draw_2dbbox_func(img, input_bbox,l2c,camera_intrinsic,name):
    output_bbox = []
    for point in input_bbox:
        project_point,depth = project_to_image(point, camera_intrinsic, l2c)
        output_bbox.append(project_point.flatten())
    output_bbox = np.array(output_bbox)
    points = output_bbox.astype(np.int32)
    # 找到最小x,y,最大x,y
    xmin = int(min(points[:,0]))
    ymin = int(min(points[:,1]))
    xmax = int(max(points[:,0]))
    ymax = int(max(points[:,1]))
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 225, 0), 3)
    # cv2.putText(img, name, tuple(points[7]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def get_2dbbox_corners(input_bbox,l2c,camera_intrinsic):
    output_bbox = []
    depth_list = []
    for point in input_bbox:
        project_point,depth = project_to_image(point, camera_intrinsic, l2c)
        output_bbox.append(project_point.flatten())
        depth_list.append(depth)
    depth_list = np.array(depth_list)
    output_bbox = np.array(output_bbox)
    points = output_bbox.astype(np.int32)
    xmin = int(min(points[:,0]))
    ymin = int(min(points[:,1]))
    xmax = int(max(points[:,0]))
    ymax = int(max(points[:,1]))
    return (xmin, ymin, xmax, ymax),depth_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim/result/standard_rcooper/standard_sim_16_rcooper_0.0_0.6')
    parser.add_argument('--save_path', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim/result/standard_rcooper/standard_sim_16_rcooper_0.0_0.6')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--road_seq', type=str, default='136-137-138-139')
    args = parser.parse_args()
    
    output_dir = os.path.join(args.save_path, args.road_seq, args.data_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cam_list = os.listdir(os.path.join(args.data_path, args.road_seq))
    cam_list = sorted([cam for cam in cam_list if cam != 'coop'])
    for cam_name in cam_list:
        output_cam_dir = os.path.join(output_dir, cam_name)
        if not os.path.exists(output_cam_dir):
            os.makedirs(output_cam_dir)
        video_path = os.path.join(output_cam_dir, 'video.avi')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # videowriter = cv2.VideoWriter(video_path,fourcc, 10.0, (1920,1200),True)
        l2c_path = os.path.join(args.data_path, args.road_seq, cam_name, 'calib', f"lidar2cam/{cam_name}.json")
        intrsinic_path = os.path.join(args.data_path, args.road_seq, cam_name, 'calib', f'camera_intrinsic/{cam_name}.json')
        with open(l2c_path, 'r') as f:
            l2c_data = json.load(f)
        l2c = np.array(l2c_data['lidar2cam'])
        with open(intrsinic_path, 'r') as f:
            camera_intrinsic_data = json.load(f)
        camera_intrinsic = np.array(camera_intrinsic_data['intrinsic'])
        
        image_dir = os.path.join(args.data_path, args.road_seq, cam_name,args.data_type,"image")
        image_list = sorted(os.listdir(image_dir))
        label_dir = os.path.join(args.data_path,args.road_seq, cam_name,args.data_type,"label/lidar_label")
        label_list = sorted(os.listdir(label_dir))
        for i in tqdm(range(len(image_list))):
            if i > 10:
                break
            img_name,ext = os.path.splitext(image_list[i])
            img_path = os.path.join(image_dir, image_list[i])
            if 'sim' in args.data_path:
                label_path = os.path.join(label_dir, image_list[i].replace(ext,'.json'))
            else:
                label_path = os.path.join(label_dir, label_list[i])
            output_path = os.path.join(output_cam_dir, image_list[i])
            if not os.path.exists(label_path):
                print(f"{label_path} not exist")
                continue
            if not os.path.exists(img_path):
                print(f"{img_path} not exist")
                continue
                
            labels = json.load(open(label_path))
            
            img = cv2.imread(img_path)
            for label in labels:
                name = label['type']
                if name != "traffic_signs":
                    input_bbox = get_3dbbox_corners(label)
                    
                    draw_3dbbox_func(img, input_bbox,l2c,camera_intrinsic,name,label)
                
            # videowriter.write(img)
            cv2.imwrite(output_path, img)
                    
                    
            
        
                