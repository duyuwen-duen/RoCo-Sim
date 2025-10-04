import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import Canvas
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

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
    
    input_bbox = np.array([
        [location[0]+points[0][0], location[1]+points[0][1], location[2]+points[0][2]],
        [location[0]+points[1][0], location[1]+points[1][1], location[2]+points[1][2]],
        [location[0]+points[2][0], location[1]+points[2][1], location[2]+points[2][2]],
        [location[0]+points[3][0], location[1]+points[3][1], location[2]+points[3][2]],
        [location[0]+points[4][0], location[1]+points[4][1], location[2]+points[4][2]],
        [location[0]+points[5][0], location[1]+points[5][1], location[2]+points[5][2]],
        [location[0]+points[6][0], location[1]+points[6][1], location[2]+points[6][2]],
        [location[0]+points[7][0], location[1]+points[7][1], location[2]+points[7][2]],
    ])
    return input_bbox

def project_to_image(points,K,l2c):
    """
    transform points (x,y,z) to image (h,w,1)
    """
    points_2d = []
    for point in points:
        point = np.array(point).reshape((3, 1))
        R = l2c[:3,:3]
        T = l2c[:3,3]
        point_camera = np.dot(R, point) + T.reshape((3, 1))
        point_2d_homogeneous = np.dot(K, point_camera)
        
        point_2d = point_2d_homogeneous[:2, :] / point_2d_homogeneous[2, :]
        points_2d.append(point_2d.flatten())
    points_2d = np.array(points_2d)
    return points_2d


class Draggable3DBox:
    def __init__(self, master, img, points_list, box_ids, projection_matrix=None, intrinsic_matrix=None, extrinsic_matrix=None):
        self.master = master
        self.canvas = Canvas(master, width=img.shape[1], height=img.shape[0], bg='white')
        self.canvas.pack()
        
        self.img = img
        self.photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        self.points_list = points_list
        self.points_2d_list, self.box_ids = self.filter_boxes_outside_image(points_list, box_ids, img.shape[1], img.shape[0], projection_matrix, intrinsic_matrix, extrinsic_matrix)
        
        self.boxes = {}
        for box_id, points_2d in zip(self.box_ids, self.points_2d_list):
            self.boxes[box_id] = self.draw_box(points_2d)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Bind middle mouse button for rotation
        self.canvas.bind("<ButtonPress-2>", self.on_start_rotation)
        self.canvas.bind("<B2-Motion>", self.on_rotate)
        self.canvas.bind("<ButtonRelease-2>", self.on_release_rotation)
        
        # Bind right mouse button for different rotation
        self.canvas.bind("<ButtonPress-3>", self.on_start_vertex_move)
        self.canvas.bind("<B3-Motion>", self.on_vertex_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_release_vertex_move)
       
        self.box_positions = {}
        self.box_rotations = {}
        self.selected_box_id = None  # ID of the selected box
        self.rotation_center = None  # Center of rotation
        self.rotation_angle = 0  # Current rotation angle
        self.rotation_start_x = None
        self.rotation_start_y = None
        self.selected_line = None  # Selected line for rotation
        self.selected_vertex = None  # Selected vertex for moving
        self.vertex_move_start_x = None
        self.vertex_move_start_y = None
        
    def filter_boxes_outside_image(self, points_list, box_ids, img_width, img_height, projection_matrix=None, intrinsic_matrix=None, extrinsic_matrix=None):
        """
        Filter out boxes whose any vertex is outside the image.
        
        Args:
            points_list (list of np.array): List of 3D points for each box.
            box_ids (list): List of IDs for each box.
            img_width (int): Width of the image.
            img_height (int): Height of the image.
            projection_matrix (np.array, optional): Projection matrix.
            intrinsic_matrix (np.array, optional): Intrinsic matrix.
            extrinsic_matrix (np.array, optional): Extrinsic matrix.
        
        Returns:
            tuple: A tuple containing the filtered points_2d_list and box_ids.
        """
        filtered_points_2d_list = []
        filtered_box_ids = []
        for points, box_id in zip(points_list, box_ids):
            points_2d = project_to_image(points, intrinsic_matrix, extrinsic_matrix)
            if any(0 <= p[0] < img_width and 0 <= p[1] < img_height for p in points_2d):
                filtered_points_2d_list.append(points_2d)
                filtered_box_ids.append(box_id)

        return filtered_points_2d_list, filtered_box_ids
    
    def draw_box(self, points_2d):
        # Draw lines connecting the points
        lines = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        box = []
        for line in lines:
            x0, y0 = points_2d[line[0]]
            x1, y1 = points_2d[line[1]]
            line_id = self.canvas.create_line(points_2d[line[0]][0], points_2d[line[0]][1],
                                              points_2d[line[1]][0], points_2d[line[1]][1], fill="red", width=3)
            box.append(line_id)
            # self.canvas.create_text(x0, y0, text=str(line[0]), fill="blue")
            # self.canvas.create_text(x1, y1, text=str(line[1]), fill="blue")
        return box

    def on_start(self, event):
        # Find which box was clicked
        self.selected_box_id = None
        threshold = 5  # Distance threshold to consider a click as "close enough"
        min_distance = float('inf')
        closest_box_id = None

        for box_id, box in self.boxes.items():
            for line_id in box:
                start_x, start_y = self.canvas.coords(line_id)[0], self.canvas.coords(line_id)[1]
                end_x, end_y = self.canvas.coords(line_id)[2], self.canvas.coords(line_id)[3]
                
                distance = self.distance_point_to_line_segment(event.x, event.y, start_x, start_y, end_x, end_y)
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    closest_box_id = box_id
        
        if closest_box_id is not None:
            self.selected_box_id = closest_box_id
            self.last_x = event.x
            self.last_y = event.y
        if self.selected_box_id is not None and self.selected_box_id not in self.box_positions:
            self.box_positions[self.selected_box_id] = self.points_2d_list[self.box_ids.index(self.selected_box_id)]
    
    def distance_point_to_line_segment(self, px, py, x1, y1, x2, y2):
        """
        Calculate the shortest distance from a point (px, py) to a line segment defined by two points (x1, y1) and (x2, y2).
        """
        dx = x2 - x1
        dy = y2 - y1
        
        dot = (px - x1) * dx + (py - y1) * dy
        
        if dot < 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        if dot > dx**2 + dy**2:
            return np.sqrt((px - x2)**2 + (py - y2)**2)
        
        return np.sqrt(((px - x1)**2 + (py - y1)**2) - (dot**2) / (dx**2 + dy**2))
    
    def on_drag(self, event):
        if self.selected_box_id is not None:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            
            # Move the selected box
            for line_id in self.boxes[self.selected_box_id]:
                self.canvas.move(line_id, dx, dy)
            
            # Update the current position
            self.last_x = event.x
            self.last_y = event.y
            
            # Update the box position dictionary
            self.box_positions[self.selected_box_id] = self.box_positions[self.selected_box_id] + [dx, dy]

    def on_release(self, event):
        if self.selected_box_id is not None:
            # # Update the file with the new coordinates
            # self.update_file()
            
            # Reset the selected box index
            self.selected_box_id = None
            
    def on_start_rotation(self, event):
        # Find the closest box edge for rotation
        self.selected_box_id = None
        threshold = 5  # Distance threshold to consider a click as "close enough"
        min_distance = float('inf')
        closest_line = None
        closest_box_id = None


        for box_id, box in self.boxes.items():
            for line_id in box:
                start_x, start_y = self.canvas.coords(line_id)[0], self.canvas.coords(line_id)[1]
                end_x, end_y = self.canvas.coords(line_id)[2], self.canvas.coords(line_id)[3]
                distance = self.distance_point_to_line_segment(event.x, event.y, start_x, start_y, end_x, end_y)
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    closest_box_id = box_id
                    closest_line = line_id
        
        if closest_line is not None:
            self.selected_line = closest_line
            self.selected_box_id = closest_box_id
            self.rotation_start_x = event.x
            self.rotation_start_y = event.y
            if self.selected_box_id not in self.box_positions:
                self.box_positions[self.selected_box_id] = self.points_2d_list[self.box_ids.index(self.selected_box_id)]
            self.rotation_center = np.mean(self.box_positions[self.selected_box_id], axis=0)
            

    def on_rotate(self, event):
        if self.selected_box_id is not None and self.rotation_center is not None:
            # Calculate the rotation angle based on the drag movement around the center
            dx = event.x - self.rotation_start_x
            dy = event.y - self.rotation_start_y
            # self.rotation_angle = np.arctan2(dy, dx) * 180 / np.pi
            self.rotation_angle = np.arctan2(dy, dx)
            
            # Rotate the selected box around its center
            rotated_points_2d = self.rotate_box_around_center(self.box_positions[self.selected_box_id], self.rotation_center, self.rotation_angle)
            
            
            # Clear the existing box lines
            for line_id in self.boxes[self.selected_box_id]:
                self.canvas.delete(line_id)
            
            # Redraw the rotated box
            self.boxes[self.selected_box_id] = self.draw_box(rotated_points_2d)
            
            # Update the rotation start position
            self.rotation_start_x = event.x
            self.rotation_start_y = event.y
            self.box_positions[self.selected_box_id] = rotated_points_2d


    def on_release_rotation(self, event):
        
        if self.selected_box_id is not None:
            # Update the box rotation dictionary
            self.box_rotations[self.selected_box_id] = self.rotation_angle
            
            # Reset the rotation state
            self.rotation_center = None
            self.rotation_start_x = None
            self.rotation_start_y = None
            self.selected_line = None
            self.selected_box_id = None

    def rotate_box_around_center(self, points_2d, center, angle):
        # Rotate points around the given center using the angle
        rotated_points = []
        for point in points_2d:
            x = point[0] - center[0]
            y = point[1] - center[1]
            cos_theta = np.cos(np.radians(angle))
            sin_theta = np.sin(np.radians(angle))
            rotated_x = x * cos_theta - y * sin_theta + center[0]
            rotated_y = x * sin_theta + y * cos_theta + center[1]
            rotated_points.append([rotated_x, rotated_y])
        return np.array(rotated_points)

    def on_start_vertex_move(self, event):
        # Find which box was clicked
        self.selected_box_id = None
        threshold = 5  # Distance threshold to consider a click as "close enough"
        min_distance = float('inf')
        closest_box_id = None

        for box_id, box in self.boxes.items():
            for line_id in box:
                start_x, start_y = self.canvas.coords(line_id)[0], self.canvas.coords(line_id)[1]
                end_x, end_y = self.canvas.coords(line_id)[2], self.canvas.coords(line_id)[3]
                
                distance = self.distance_point_to_line_segment(event.x, event.y, start_x, start_y, end_x, end_y)
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    closest_box_id = box_id
        self.selected_box_id = closest_box_id
        if self.selected_box_id is not None:
            self.selected_vertex = None
            min_distance = float('inf')
            closest_vertex_index = None

            # Find the closest vertex to the mouse click
            for i, point in enumerate(self.box_positions[self.selected_box_id]):
                distance = np.sqrt((event.x - point[0])**2 + (event.y - point[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_vertex_index = i

            if closest_vertex_index is not None:
                self.selected_vertex = closest_vertex_index
                self.vertex_move_start_x = event.x
                self.vertex_move_start_y = event.y

    def on_vertex_move(self, event):
        if self.selected_box_id is not None and self.selected_vertex is not None:
            dx = event.x - self.vertex_move_start_x
            dy = event.y - self.vertex_move_start_y

            # Update the selected vertex's position
            self.box_positions[self.selected_box_id][self.selected_vertex][0] += dx
            self.box_positions[self.selected_box_id][self.selected_vertex][1] += dy

            # Update the box drawing
            self.redraw_box(self.selected_box_id)

            # Update the start position for the next move
            self.vertex_move_start_x = event.x
            self.vertex_move_start_y = event.y

    def on_release_vertex_move(self, event):
        if self.selected_box_id is not None and self.selected_vertex is not None:

            # Reset the selected vertex index
            self.selected_vertex = None
            self.selected_box_id = None

    def redraw_box(self, box_id):
        # Clear the existing box lines
        for line_id in self.boxes[box_id]:
            self.canvas.delete(line_id)

        # Redraw the updated box
        self.boxes[box_id] = self.draw_box(self.box_positions[box_id])
    

def update_file(file,box_positions):
    # Write the coordinates to the file
    file.seek(0)  # Go to the beginning of the file
    file.truncate()  # Clear the file
    for box_id, points in box_positions.items():
        formatted_coordinates = str(points).replace('\n', '')
        file.write(f"BoxID {box_id} {formatted_coordinates}\n")
    file.flush()  # Ensure the data is written to the file immediately

def draw_bbox(image_list, label_list, scene_folder,intrinsic_matrix,extrinsic_matrix,name):
    
    for i, (image_file_path, label_file_path) in enumerate(zip(image_list, label_list)):
        img = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        labels_json = json.load(open(label_file_path))
        
        
        for label in labels_json:
            points_3d = get_3dbbox_corners(label)
            points_2d = project_to_image(points_3d, intrinsic_matrix, extrinsic_matrix)
            
            idx = 0
            if any(0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0] for p in points_2d):
                for point in points_2d:
                    cv2.circle(img, tuple(int(x) for x in point), 1, (225, 50, 0), -1)  # 绘制点
                    cv2.putText(img,str(idx),tuple(int(x) for x in point),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,0,255),2)
                    idx += 1
                    
                for j in range(0, 4):
                    next_idx = (j + 1) % 8
                    cv2.line(img, tuple(int(x) for x in points_2d[j]), tuple(int(x) for x in points_2d[next_idx%4]), (255, 225, 0), 2)
                    cv2.line(img, tuple(int(x) for x in points_2d[j+4]), tuple(int(x) for x in points_2d[next_idx%4+4]), (255, 225, 0), 2)

                for j in range(0, 4):
                    opposite_idx = j + 4
                    cv2.line(img, tuple(int(x) for x in points_2d[j]), tuple(int(x) for x in points_2d[opposite_idx]), (255, 225, 0), 2)
                
    
        output_image_path = os.path.join(scene_folder,name, os.path.basename(image_file_path))
        if not os.path.exists(os.path.dirname(output_image_path)):
            os.makedirs(os.path.dirname(output_image_path))
        cv2.imwrite(output_image_path, img)


def process_images(image_list, label_list, scene_folder,extrinsic_matrix, intrinsic_matrix):
    if os.path.exists(scene_folder):
        response = input(f"The folder '{scene_folder}' already exists. Do you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
    
    os.makedirs(scene_folder, exist_ok=True)
    # Initialize the file to store coordinates
    all_box_positions = {}
    original_3d_boxes ={}
    
    # Process each image and label pair
    for i, (image_file_path, label_file_path) in enumerate(zip(image_list, label_list)):
        img = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        labels_json = json.load(open(label_file_path))
        
        # Initialize the GUI
        root = tk.Tk()
        root.title("Draggable 3D Box")
        
        # Process the boxes in the label file
        
        points_list = []
        box_ids = []
        original_3d_box = {}
        for idx,label in enumerate(labels_json):
            idx = image_file_path.split("/")[-1].split(".")[0]+'-'+label['type']+'-'+str(idx)
            points_3d = get_3dbbox_corners(label)
            box_ids.append(idx)
            points_list.append(points_3d)
            original_3d_box[idx] = label
        
        app = Draggable3DBox(root, img, points_list, box_ids, projection_matrix, intrinsic_matrix, extrinsic_matrix)
        
        # Handle keyboard events
        def handle_keypress(event):
            if event.keysym == "space":
                root.destroy()
        
        root.bind_all("<KeyPress>", handle_keypress)
        
        # Run the main loop
        root.mainloop()
        
        # Save the box positions
        for box_id, points in app.box_positions.items():
            all_box_positions[box_id] = points
            original_3d_boxes[box_id] = original_3d_box[box_id]

    appfile = open(os.path.join(scene_folder, 'box_coordinates.txt'), 'w')
    update_file(appfile, all_box_positions)
    appfile.close()

    return all_box_positions, original_3d_boxes


def orthogonalize(matrix):
    u, _, vh = np.linalg.svd(matrix[:3, :3])
    return np.dot(u, vh)

def is_orthogonal(matrix):
    mat = np.array(matrix)
    transpose = mat.T
    product = np.dot(mat, transpose)
    return np.allclose(product, np.eye(mat.shape[0]))

def error_loss(extrinsic_matrix, correct_box_positions, original_3d_boxes,intrinsic_matrix):
    error = 0
    extrinsic_matrix = np.array(extrinsic_matrix).reshape(4,4)
    extrinsic_matrix[:3,:3] = orthogonalize(extrinsic_matrix[:3,:3])
    for key in correct_box_positions.keys():
        
        correct_points = correct_box_positions[key]
        label = original_3d_boxes[key]
        points_3d = get_3dbbox_corners(label)
        points_2d = project_to_image(points_3d, intrinsic_matrix, extrinsic_matrix)
        ratio = abs(points_2d[0][0]-correct_points[1][0])
        error += np.linalg.norm(correct_points - points_2d)/np.sqrt(ratio)

    print(f"loss: {error}")                
    return error


def load_projection_matrix():
    return None
    
def load_intrinsic_and_extrinsic(calib_path,cam_id=0):
    with open(calib_path) as f:
        data = json.load(f)
        intrinsic_matrix = np.array(data[f'cam_{str(cam_id)}']['intrinsic'])
        extrinsic_matrix = np.array(data[f'cam_{str(cam_id)}']['extrinsic'])
    extrinsic_matrix[:3,:3] = orthogonalize(extrinsic_matrix[:3,:3])
    return intrinsic_matrix, extrinsic_matrix

if __name__ == '__main__':

    road_name = "117"
    data_root = "/home/ubuntu/duyuwen/RoCo-Sim/roco_sim/calib_tools/demo_data"
    cam_id = 0
    calib_path = os.path.join(data_root, f"data/{road_name}/calib/{road_name}.json")
    scene_folder = os.path.join(data_root, f"result/{road_name}")
    image_folder = os.path.join(data_root, f"data/{road_name}/image")
    image_list = sorted(os.listdir(image_folder))
    image_list = [os.path.join(image_folder, data_path) for data_path in image_list]
    label_path = os.path.join(data_root, f"data/{road_name}/label")
    label_list = sorted(os.listdir(label_path))
    label_list = [os.path.join(label_path, data_path) for data_path in label_list]


    projection_matrix = load_projection_matrix()
    intrinsic_matrix, extrinsic_matrix = load_intrinsic_and_extrinsic(calib_path)
    draw_bbox(image_list, label_list, scene_folder,extrinsic_matrix=extrinsic_matrix, intrinsic_matrix=intrinsic_matrix, name="origin_3dbbox")
    correct_box_positions, original_3d_boxes = process_images(image_list, label_list, scene_folder,extrinsic_matrix=extrinsic_matrix, intrinsic_matrix=intrinsic_matrix)
    result = minimize(error_loss,extrinsic_matrix.ravel(),args=(correct_box_positions,original_3d_boxes,intrinsic_matrix),method='L-BFGS-B')
    new_extrinsic = result.x.reshape(4,4)
    print(f'origin extrinsic_matrix:{extrinsic_matrix}')
    # print(result.x)
    print(f'is_orthogonal:{is_orthogonal(new_extrinsic[:3,:3])}')
    print(f'new extrinsic_matrix:{new_extrinsic}')
    new_extrinsic[:3,:3] = orthogonalize(new_extrinsic[:3,:3])
    print(f'is orthogonalized:{is_orthogonal(new_extrinsic[:3,:3])}')
    print(f'new extrinsic_matrix:{new_extrinsic}')

    json_data = json.dumps(new_extrinsic.tolist(), indent=4)
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    calib_data[f'cam_{cam_id}']['extrinsic'] = new_extrinsic.tolist()
        
    draw_bbox(image_list, label_list, scene_folder,intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=new_extrinsic,name='new_3dbbox')

    if not os.path.exists(os.path.join(scene_folder, 'calib')):
        os.makedirs(os.path.join(scene_folder, 'calib'))
    save_calib_path = os.path.join(scene_folder, f'calib/{road_name}.json')
    
    with open(save_calib_path, 'w') as f:
        json.dump(calib_data, f, indent=4)
                