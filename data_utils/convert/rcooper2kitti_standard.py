import argparse
import os
import numpy as np
from label_tools import *
import cv2
from tqdm import tqdm

color_map = {"Car":(0, 0, 255), "Bus":(0, 0, 255), "Pedestrian":(0, 0, 255), "Cyclist":(0, 0, 255),'Truck':(0, 0, 255)}
color_gt_map = {"Car":(0, 255, 0), "Bus":(0, 255, 0), "Pedestrian":(0, 255, 0), "Cyclist":(0, 255, 0),'Truck':(0, 255, 0)}
ego_dict = {
    '106-105':'106-1',
    '116-115':'116-0',
    '117-118-120-119':'117-0',
    '136-137-138-139':'136-0' ,
    'intersection':'south1-0'
}
def project_to_image(pts_3d, P):
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  return pts_2d

def draw_box_3d(image, corners, c=(0, 255, 0)):
  face_idx = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
  for ind_f in [3, 2, 1, 0]:
    f = face_idx[ind_f]
    for j in [0, 1, 2, 3]:
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
               (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
               (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
  return image

def get_cam_8_points_label(labels, r_velo2cam, t_velo2cam):

    camera_8_points_list = []
    h, w, l, x, y, z, yaw_lidar = labels
    z = z - h / 2
    bottom_center = [x, y, z]
    obj_size = [l, w, h]
    # lidar_8_points = compute_box_3d(obj_size, bottom_center, -yaw_lidar-np.pi*0.5)
    lidar_8_points = compute_box_3d(obj_size, bottom_center, yaw_lidar)
    camera_8_points = r_velo2cam * np.matrix(lidar_8_points).T + t_velo2cam.reshape(3,-1)
    camera_8_points_list.append(camera_8_points.T)
    return camera_8_points_list

def draw_3d_box_on_image(image, label_2_file, P2, denorm, c=(0, 255, 0),gt=False):
    with open(label_2_file) as f:
      for line in f.readlines():
          line_list = line.split('\n')[0].split(' ')
          object_type = line_list[0]
          if object_type not in color_map.keys(): continue
          dim = np.array(line_list[8:11]).astype(float)
          location = np.array(line_list[11:14]).astype(float)
          rotation_y = float(line_list[14])
          box_3d = compute_box_3d_camera(dim, location, rotation_y, denorm)
          box_2d = project_to_image(box_3d, P2)
          if gt:
              image = draw_box_3d(image, box_2d, c=color_gt_map[object_type])
          else:
              image = draw_box_3d(image, box_2d, c=color_map[object_type])
    return image

def compute_box_3d_camera(dim, location, rotation_y, denorm):
    # rotation_y =  - rotation_y
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    # l, w, h = dim[1], dim[2], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) 

    denorm = denorm[:3]
    denorm_norm = denorm / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    ori_denorm = np.array([0.0, -1.0, 0.0])
    theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
    n_vector = np.cross(denorm, ori_denorm)
    n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
    corners_3d = np.dot(rotation_matrix, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(Tr_velo_to_cam):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(Tr_velo_to_cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

parser = argparse.ArgumentParser("Generate the Kitti Format Label")
parser.add_argument("--data_path", type=str, default="/home/ubuntu/duyuwen/RoCo-Sim/data/standard_tum", help="Raw data root about RCooper_standard.")
parser.add_argument("--demo", type=bool, default=False, help="Visualize the label or not.")
parser.add_argument("--demo_path", type=str, default="/home/ubuntu/duyuwen/RoCo-Sim/result/vis", help="The path to save the visualized label.")

if __name__ == "__main__":
    print("================ Start to Convert ================")
    args = parser.parse_args()
    root_path = args.data_path
    vis_judge = args.demo
    vis_path = args.demo_path
    road_list = os.listdir(root_path)
    print(f'road_list: {road_list}')

    for road in road_list:
        if '.pkl' in road:
            continue
        road_path = os.path.join(root_path, road)
        camera_list = os.listdir(road_path)
        print(f'camera_list: {camera_list}')
        ego_cam = ego_dict[road]
        #115-0,115-1.etc
        for camera in camera_list:
            camera_path = os.path.join(road_path, camera)
            if camera == 'coop':
                calib_lidar2cam = read_json(
                    os.path.join(os.path.join(road_path, ego_cam), "calib", "lidar2cam", ego_cam+'.json'))
            else:
                calib_lidar2cam = read_json(
                    os.path.join(camera_path, "calib", "lidar2cam", camera + '.json'))
            r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
            Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))
            if camera == 'coop':
                calib_intrinsic = read_json(os.path.join(os.path.join(road_path, ego_cam), "calib", "camera_intrinsic", ego_cam+'.json'))
            else:
                calib_intrinsic = read_json(os.path.join(camera_path, "calib", "camera_intrinsic", camera+'.json'))
            P2 = get_P(calib_intrinsic)
            data_name = ['train', 'val']
            for data in data_name:
                if 'sim' in args.data_path:
                    if data == 'val':
                        continue
                camera_label_path = os.path.join(camera_path, data, 'label', 'camera_label')
                if camera == 'coop':
                    camera_label_path = os.path.join(camera_path, data, 'label', 'lidar_label')
                kitti_label_path = os.path.join(camera_path, data, 'label', 'kitti_label')
                mkdir_p(kitti_label_path)
                camera_labels = os.listdir(camera_label_path)
                for camera_label in camera_labels:
                    labels = read_json(os.path.join(camera_label_path, camera_label))
                    for label in labels:
                        h, w, l, x, y, z, yaw_lidar = get_label(label)
                        z = z - h / 2
                        bottom_center = [x, y, z]
                        obj_size = [l, w, h]

                        bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam

                        alpha, yaw = get_camera_3d_8points(
                        obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
                        )
                        cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

                        set_label(label, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)
                    os.path.splitext(camera_label)[0] + ".txt"
                    write_path = os.path.join(kitti_label_path, os.path.splitext(camera_label)[0] + ".txt")
                    write_kitti_in_txt(labels, write_path)
                rewrite_label(kitti_label_path)
                label_filter(kitti_label_path)
    
                if vis_judge:
                    kitti_labels = os.listdir(kitti_label_path)
                    new_labels = []
                    for label in kitti_labels:
                      new_label = label.replace('_', '.')
                      new_labels.append(new_label)
                    new_labels = sorted(new_labels)
                    for i in range(len(new_labels)):
                      label = new_labels[i]
                      name, ext = os.path.splitext(label)
                      new_name = name.replace('.', '_')
                      new_label = new_name + ext
                      new_labels[i] = new_label
                    image_path = os.path.join(camera_path, data, 'image')
                    images = os.listdir(image_path)
                    new_images = []
                    for image in images:
                      new_image = image.replace('_', '.')
                      new_images.append(new_image)
                    new_images = sorted(new_images)
                    for i in range(len(new_images)):
                      image = new_images[i]
                      name, ext = os.path.splitext(image)
                      new_name = name.replace('.', '_')
                      new_image = new_name + ext
                      new_images[i] = new_image
                      
                    for i, kitti_label in enumerate(new_labels):
                        label_2_file = os.path.join(kitti_label_path, kitti_label)
                        T_velo2cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
                        image_path = os.path.join(camera_path, data, 'image')
                        image_path = os.path.join(image_path, new_images[i])
                        image = cv2.imread(image_path)
                        denorm = get_denorm(T_velo2cam)
                        image = draw_3d_box_on_image(image, label_2_file, P2, denorm, gt=True)
                        result_path = os.path.join(vis_path, road, camera, data)
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)
                        cv2.imwrite(os.path.join(result_path,kitti_label.replace('txt', 'jpg')), image)
                        

    print("================ Finish Convert ================")