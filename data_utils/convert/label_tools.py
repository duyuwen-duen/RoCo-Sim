import argparse
import os
import json
import numpy as np
import math
import errno

def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def compute_corners_3d(dim, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    # R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[0], dim[1], dim[2]
    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2]
    z_corners = [h, h, 0, 0, h, h, 0, 0]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners).transpose(1, 0)
    return corners_3d

def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    corners_3d = compute_corners_3d(dim, rotation_y)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
    return corners_3d

def convert_point(point, matrix):
    return matrix @ point

def get_cam_8_points(bboxes,r_velo2cam, t_velo2cam):
    """
    Args:
        bbox: [x, y, z, l, w, h, lidar_yaw]
    Returns:
    """
    camera_8_points_list = []
    for bbox in range(bboxes.shape[0]):
        x, y, z, l, w, h, yaw_lidar = bboxes[bbox]
        z = z - h / 2
        bottom_center = [x, y, z]
        obj_size = [l, w, h]
        lidar_8_points = compute_box_3d(obj_size, bottom_center, yaw_lidar)
        camera_8_points = r_velo2cam * np.matrix(lidar_8_points).T + t_velo2cam
        camera_8_points_list.append(camera_8_points.T)
    return camera_8_points_list

def get_lidar2cam(calib):
    r_velo2cam = np.array([row[:3] for row in calib["lidar2cam"][:3]])
    t_velo2cam = np.array([row[3] for row in calib["lidar2cam"][:3]])
    r_velo2cam = r_velo2cam.reshape((3, 3))
    t_velo2cam = t_velo2cam.reshape((3, 1))
    return r_velo2cam, t_velo2cam

def get_P(calib):
    intrinsic = calib['intrinsic']
    intrinsic_matrix = np.array(intrinsic)
    P2 = np.hstack((intrinsic_matrix, np.zeros((3, 1))))
    return P2

def get_label(label):
    h = float(label["3d_dimensions"]["h"])
    w = float(label["3d_dimensions"]["w"])
    length = float(label["3d_dimensions"]["l"])
    x = float(label["3d_location"]["x"])
    y = float(label["3d_location"]["y"])
    z = float(label["3d_location"]["z"])
    rotation_y = float(label["rotation"])
    return h, w, length, x, y, z, rotation_y

def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam

    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)

    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])

    # add transfer
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi

    alpha_arctan = normalize_angle(alpha)

    return alpha_arctan, yaw

def normalize_angle(angle):
    # make angle in range [-0.5pi, 1.5pi]
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def points_cam2img(points_3d, calib_intrinsic, with_depth=False):
    """Project points from camera coordicates to image coordinates.

    points_3d: N x 8 x 3
    calib_intrinsic: 3 x 4
    return: N x 8 x 2
    """
    calib_intrinsic = np.hstack((calib_intrinsic,np.array([[0],[0],[0]])))
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0)
    points_2d_shape = np.concatenate([points_num, [3]], axis=0)
    # previous implementation use new_zeros, new_one yeilds better results

    points_4 = np.concatenate((points_3d, np.ones(points_shape)), axis=-1)
    point_2d = np.matmul(calib_intrinsic, points_4.T.swapaxes(1, 2).reshape(4, -1))
    point_2d = point_2d.T.reshape(points_2d_shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    point_2d_res = (point_2d_res - 1).round()
    if with_depth:
        return np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)
    return point_2d_res
def points_cam2img_center(points_3d, calib_intrinsic, with_depth=False):
    """Project points from camera coordicates to image coordinates.

    points_3d: N  x 3
    calib_intrinsic: 3 x 4
    return: N  x 2
    """
    calib_intrinsic = np.hstack((calib_intrinsic,np.array([[0],[0],[0]])))
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0)
    points_2d_shape = np.concatenate([points_num, [3]], axis=0)
    # previous implementation use new_zeros, new_one yeilds better results

    points_4 = np.concatenate((points_3d, np.ones(points_shape)), axis=-1)
    point_2d = np.matmul(calib_intrinsic, points_4.T)
    point_2d = point_2d.T.reshape(points_2d_shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    point_2d_res = (point_2d_res - 1).round()
    if with_depth:
        return np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)
    return point_2d_res

def check_bbox_in_image_center(bboxes_center, image_size=[1200,1920]):

    height, width = image_size
    num_box = bboxes_center.shape[0]
    masks = np.zeros(num_box, dtype=bool)

    for i in range(num_box):
        box = bboxes_center[i]
        x_coords = box[0]
        y_coords = box[1]
        if np.all((x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)):
            masks[i] = True

    return masks

def get_annos_all(path):
    my_json = read_json(path)
    gt_names = []
    gt_boxes = []
    other_infos = []
    for item in my_json:
        gt_names.append(item["type"].lower())
        x, y, z = float(item["3d_location"]["x"]), float(item["3d_location"]["y"]), float(item["3d_location"]["z"])
        h, w, l = float(item["3d_dimensions"]["h"]), float(item["3d_dimensions"]["w"]), float(item["3d_dimensions"]["l"])                                                            
        lidar_yaw = float(item["rotation"])
        gt_boxes.append([x, y, z, l, w, h, lidar_yaw])
        occuluded_state = item["occluded_state"]
        truncated_state = item["truncated_state"]
        crowding = item["crowding"]
        ignore = item["ignore"]
        track_id = item["track_id"]
        other_infos.append([occuluded_state, truncated_state, crowding,
                            ignore, track_id])
    gt_boxes = np.array(gt_boxes)
    other_infos = np.array(other_infos)
    return gt_names, gt_boxes, other_infos

def save_annos(gt_names, gt_boxes, other_infos):
    annos = []
    for name, box, info in zip(gt_names, gt_boxes, other_infos):
        x, y, z, l, w, h, lidar_yaw = box
        occuluded_state, truncated_state, crowding, ignore, track_id = info
        item = {
            "type": name,
            "occluded_state": int(occuluded_state),
            "truncated_state": int(truncated_state),
            "crowding": int(crowding),
            "ignore": int(ignore),
            "track_id": int(track_id),  
            "3d_location": {
                "x": x,
                "y": y,
                "z": z
            },
            "3d_dimensions": {
                "w": w,
                "h": h,
                "l": l
            },
            "rotation": lidar_yaw
        }
        # print(item)
        annos.append(item)
    return annos


def filter_out_img_box_center_new(gt_names,gt_bboxes, other_infos, calib_intrinsic, r_velo2cam, t_velo2cam):
    """
    """
    new_gt_names = []
    new_gt_bboxes = []
    new_other_infos = []
    camera_8_points_list = get_cam_8_points(gt_bboxes, r_velo2cam, t_velo2cam)
    cam8points = np.array(camera_8_points_list)
    num_bbox = cam8points.shape[0]
    if num_bbox == 0:
        return new_gt_names,new_gt_bboxes, new_other_infos
    else:
        center_point = np.average(cam8points, axis=1)
        uv_origin = points_cam2img_center(center_point, calib_intrinsic)
        uv_origin = (uv_origin - 1).round()
        mask = check_bbox_in_image_center(uv_origin)
        new_gt_names = np.array(gt_names)[mask]
        new_gt_bboxes = gt_bboxes[mask]
        new_other_infos = other_infos[mask]
        return new_gt_names,new_gt_bboxes, new_other_infos

def write_kitti_in_txt(my_json, path_txt):
    wf = open(path_txt, "w")
    for item in my_json:
        i1 = str(item["type"]).title()
        i2 = str(item["truncated_state"])
        i3 = str(item["occluded_state"])
        i4 = str(item["alpha"])
        # i5, i6, i7, i8 = (
        #     str(item["2d_box"]["xmin"]),
        #     str(item["2d_box"]["ymin"]),
        #     str(item["2d_box"]["xmax"]),
        #     str(item["2d_box"]["ymax"]),
        # )
        i5, i6, i7, i8 = (
            str(0),
            str(0),
            str(0),
            str(0),
        )
        # i9, i11, i10 = str(item["3d_dimensions"]["h"]), str(item["3d_dimensions"]["w"]), str(item["3d_dimensions"]["l"])
        i9, i11, i10 = str(item["3d_dimensions"]["h"]), str(item["3d_dimensions"]["l"]), str(item["3d_dimensions"]["w"])
        i12, i13, i14 = str(item["3d_location"]["x"]), str(item["3d_location"]["y"]), str(item["3d_location"]["z"])
        # i15 = str(float(item["rotation"]))
        i15 = str(float(item["rotation_y"]))
        # i15 = str(-0.5 * np.pi  - float(item["rotation"]))
        item_list = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
        item_string = " ".join(item_list) + "\n"
        wf.write(item_string)
    wf.close()



def set_label(label, h, w, length, x, y, z, alpha, rotation_y):
    label["3d_dimensions"]["h"] = h
    label["3d_dimensions"]["w"] = w
    label["3d_dimensions"]["l"] = length
    label["3d_location"]["x"] = x
    label["3d_location"]["y"] = y
    label["3d_location"]["z"] = z
    label["alpha"] = alpha
    label["rotation_y"] = rotation_y
    # label["2d_box"] = None
    label["2d_box"] = None

def rewrite_txt(path):
    with open(path, "r+") as f:
        data = f.readlines()
        find_str1 = "Truck"
        find_str2 = "Van"
        find_str3 = "Bus"
        find_str4 = "Tricyclist"
        find_str5 = "Motorcyclist"
        find_str6 = "Barrowlist"
        find_str7 = 'Bicycle'
        find_str8 = 'Tricycle'
        find_str9 = 'Motorcycle'
        find_str10 = 'Huge_Vehicle'
        replace_str1 = "Car"
        replace_str2 = "Cyclist"
        new_data = ""
        for line in data:
            if find_str1 in line:
                line = line.replace(find_str1, replace_str1)
            if find_str2 in line:
                line = line.replace(find_str2, replace_str1)
            if find_str3 in line:
                line = line.replace(find_str3, replace_str1)
            if find_str4 in line:
                line = line.replace(find_str4, replace_str2)
            if find_str5 in line:
                line = line.replace(find_str5, replace_str2)
            if find_str6 in line:
                line = line.replace(find_str6, replace_str2)
            if find_str7 in line:
                line = line.replace(find_str7, replace_str2)
            if find_str8 in line:
                line = line.replace(find_str8, replace_str2)
            if find_str9 in line:
                line = line.replace(find_str9, replace_str2)
            if find_str10 in line:
                line = line.replace(find_str10, replace_str1)
            new_data = new_data + line
    os.remove(path)
    f_new = open(path, "w")
    f_new.write(new_data)
    f_new.close()

def get_files_path(path_my_dir, extention=".json"):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(path_my_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == extention:
                path_list.append(os.path.join(dirpath, filename))
    return path_list

def rewrite_label(path_file):
    path_list = get_files_path(path_file, ".txt")
    for path in path_list:
        rewrite_txt(path)

def label_filter(label_dir):
    label_dir = label_dir
    files = os.listdir(label_dir)

    for file in files:
        path = os.path.join(label_dir, file)

        lines_write = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                wlh = float(line.split(" ")[9])
                if wlh > 0:
                    lines_write.append(line)

        with open(path, "w") as f:
            f.writelines(lines_write)
