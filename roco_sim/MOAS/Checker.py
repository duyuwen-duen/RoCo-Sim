import os
import json
import numpy as np
import sys
sys.path.append(f"{os.path.dirname(__file__)}/../vis")
from draw_bbox import get_3dbbox_corners,project_to_image,get_2dbbox_corners
import copy
import random
import argparse
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


class Checker:
    def __init__(self,H,W):
        self.occupancy_map = []
        self.occupancy_map_init = []
        self.H = H
        self.W = W
        
    def convert_label_to_lidar(self,item_origin,origin_l2w,target_l2w):
        item = copy.deepcopy(item_origin)
        location_lidar = np.linalg.inv(target_l2w) @ origin_l2w @ np.array([item['3d_location']['x'],item['3d_location']['y'],item['3d_location']['z'],1]).T
        item['3d_location']['x'] = location_lidar[0]
        item['3d_location']['y'] = location_lidar[1]
        item['3d_location']['z'] = location_lidar[2]
        theta = item['rotation']
            
        R_lidar = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        R_world = np.dot(origin_l2w[:3,:3],R_lidar)
        R_lidar2 = np.dot(np.linalg.inv(target_l2w[:3,:3]),R_world)
        euler_angles = R.from_matrix(R_lidar2).as_euler('xyz', degrees=False)
        item['rotation'] = euler_angles[2]
        return item
        
    def check_in_image(self,box,cam_id):
        box_list = []
        xmin,ymin,xmax,ymax = box
        box_list.append([xmin,ymin])
        box_list.append([xmax,ymin])
        box_list.append([xmax,ymax])
        box_list.append([xmin,ymax])
        for point in box_list:
            if point[0] >= 0 and point[0] < self.W[cam_id] and point[1] >= 0 and point[1] < self.H[cam_id]:
                return True
        return False
        
    def NMS(self, box1,box2,cam_id):
        box1 = list(box1)
        box2 = list(box2)
        if (self.check_in_image(box1,cam_id) and self.check_in_image(box2,cam_id)):
            box1[0] = max(box1[0],0)
            box1[1] = max(box1[1],0)
            box1[2] = min(box1[2],self.W[cam_id])
            box1[3] = min(box1[3],self.H[cam_id])
            inter_xmin = np.maximum(box1[0], box2[0])
            inter_ymin = np.maximum(box1[1], box2[1])
            inter_xmax = np.minimum(box1[2], box2[2])
            inter_ymax = np.minimum(box1[3], box2[3])

            inter_width = inter_xmax - inter_xmin
            inter_height = inter_ymax - inter_ymin

            if inter_width > 0 and inter_height > 0:
                intersection_area = inter_width * inter_height
            else:
                intersection_area = 0
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            min_area = min(box1_area,box2_area)
            max_area = max(box1_area,box2_area)
            union_area = min_area
            if union_area != 0:
                iou = intersection_area / union_area
                return iou
            else:
                return 0
        else:
            return 0


    def init_occupancy_map(self,labels):
        self.occupancy_map_init = []
        for label in labels:
            bbox = get_3dbbox_corners(label)
            self.occupancy_map_init.append(bbox)

    def expand_rectangle_2d(self,corners, scale=0.3):
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        diag_vector = np.array([max_x - min_x, max_y - min_y])
        diag_length = np.linalg.norm(diag_vector)
        expanded_diag_length = diag_length * scale
        expanded_diag_vector = diag_vector / diag_length * expanded_diag_length
        expanded_corners = np.array([
            min_x - expanded_diag_vector[0] / 2, min_y - expanded_diag_vector[1] / 2,
            max_x + expanded_diag_vector[0] / 2, max_y + expanded_diag_vector[1] / 2
        ])
        return expanded_corners

    def check_intersection(self, corners1, corners2):
        corners1_xy = corners1[:, :2]
        corners2_xy = corners2[:, :2]
        expanded_corners1 = self.expand_rectangle_2d(corners1_xy)
        expanded_corners2 = self.expand_rectangle_2d(corners2_xy)
        min_x1, min_y1, max_x1, max_y1 = expanded_corners1
        min_x2, min_y2, max_x2, max_y2 = expanded_corners2
        return (min_x1 <= max_x2 and min_x2 <= max_x1 and
                min_y1 <= max_y2 and min_y2 <= max_y1)

    def check_occupancy(self,car_info,target_pos):
        item = copy.deepcopy(target_pos)
        item["3d_location"]["z"] = item["3d_location"]["z"] - item["3d_dimensions"]["h"]/2 + car_info["h"]/2
        item['3d_dimensions']['h'] = car_info["h"]
        item['3d_dimensions']['w'] = car_info["w"]
        item['3d_dimensions']['l'] = car_info["l"]
        target_bbox = get_3dbbox_corners(item)
        
        for bbox in self.occupancy_map:
            if self.check_intersection(target_bbox,bbox) == True:
                return False
        return True

    def check_place_available(self,car_info,target_pos,labels,camera_info,base_road,insert_range):
        camera_intrinsic,l2cs,l2ws = camera_info
        item = copy.deepcopy(target_pos)
        item["3d_location"]["z"] = item["3d_location"]["z"] - item["3d_dimensions"]["h"]/2 + car_info["h"]/2
        item['3d_dimensions']['h'] = car_info["h"]
        item['3d_dimensions']['w'] = car_info["w"]
        item['3d_dimensions']['l'] = car_info["l"]
        
        output_bboxs = []
        depth_list = []
        for cam_id in range(len(l2cs)):
            input_bbox = get_3dbbox_corners(item)
            # convert to the lidar coordinate of the camera
            input_bbox = np.linalg.inv(l2ws[cam_id]) @ l2ws[base_road] @ np.array([[x,y,z,1] for x,y,z in input_bbox]).T
            input_bbox = np.array([[x,y,z] for x,y,z,_ in (input_bbox/input_bbox[3]).T])
            (xmin,ymin,xmax,ymax),depth = get_2dbbox_corners(input_bbox,l2cs[cam_id],camera_intrinsic[cam_id])
            output_bboxs.append(np.array([xmin,ymin,xmax,ymax]))
            depth_list.append(depth)
        depth_list = np.array(depth_list)
            
        available_flag = True
        threshold_min,threshold_max = insert_range
        max_iou = 0
        for i in range(len(labels)):
            for cam_id in range(len(l2cs)):
                label = self.convert_label_to_lidar(labels[i],l2ws[base_road],l2ws[cam_id])
                bbox_2d_points,depth = get_2dbbox_corners(get_3dbbox_corners(label),l2cs[cam_id],camera_intrinsic[cam_id])
                target_bbox_2d = [output_bboxs[cam_id][0],output_bboxs[cam_id][1],output_bboxs[cam_id][2],output_bboxs[cam_id][3]]
                if all(depth < 1) or all(depth_list[cam_id] < 1):
                    continue 
                iou = self.NMS(bbox_2d_points,target_bbox_2d,cam_id)
                if (iou > threshold_max):
                    available_flag = False
                    break
                if iou > max_iou:
                    max_iou = iou
                    
            if available_flag == False:
                break
                    
        if max_iou < threshold_min:
            available_flag = False
                    
        return available_flag,item
