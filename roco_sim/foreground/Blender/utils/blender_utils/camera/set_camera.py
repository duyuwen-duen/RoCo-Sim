"""
Set up 
i) camera intrinsic
ii) camera extrinsic
"""
import numpy as np
import pyquaternion
import bpy
import json
from mathutils import Matrix
import os


default_sensor_height = 9.6 # set to 24 mm
default_sensor_width = 12.8


# camera_rot_path = "/home/ubuntu/duyuwen/ChatSim_v2x/result/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure-example/info/c2v.json"
# with open(camera_rot_path, 'r') as f:
#     camera_info = json.load(f)
# camera_rot = camera_info['wxyz']
# camera_loc = camera_info['loc']
# camera_loc[2] = 6.3233
# print("camera_rot", camera_rot)
# print("camera_loc", camera_loc)
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: 
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    print(K)
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    R_world2bcam = cam.rotation_quaternion.to_matrix().transposed()
    T_world2bcam = -1*(R_world2bcam @ cam.location)
   
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    print(RT)
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT


def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))



def get_rotation_quaternion(ext):
    # Inputs:
    # ext: np.ndarry [4,4]
    # should first flip camera coodinate to Blender camera. x (right) y (down) z (front) -> x (right) y (up) z (back).
    # Returns:
    # w,x,y,z, where w + xi + yj + zk
    flip_matrix = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
    ext = ext @ flip_matrix
    q = pyquaternion.Quaternion(matrix=ext)
    return [q.w, q.x, q.y, q.z]

def get_location(ext):
    return ext[:3, 3]

def get_focal_in_mm(W_pixel, focal_pixel):
    # default sensor fit H is 24mm
    focal_mm = default_sensor_width / W_pixel * focal_pixel
    
    return focal_mm

def set_camera_params(intrinsic, cam2world, camera_obj_name="Camera"):
    """
    Args:
        int: Dict
            {H: xx, W:xx, focal:xx}
        cam2world: np.ndarray, OpenCV coordinate system for camera.
            shape [4,4]
        camera_obj_name: str
            name of Camera
    """
    # create a new camera object
    if camera_obj_name not in bpy.data.objects:
        # remove exising camera
        cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
        for camera in cameras:
            bpy.data.objects.remove(camera, do_unlink=True)

        bpy.ops.object.camera_add(enter_editmode=False)
        
        if hasattr(bpy.context, 'object'):
            camera = bpy.context.object
            camera.name = camera_obj_name
        else: # if-statement will raise error when not in background mode.
            camera = bpy.data.objects['Camera']

        # activate the camera
        bpy.context.scene.camera = camera

    focal_in_mm = get_focal_in_mm(intrinsic['W'], intrinsic['focal_x'])
    camera = bpy.data.objects[camera_obj_name] 
    camera.location = get_location(cam2world)
    camera.rotation_mode = 'QUATERNION'
    camera_rot = get_rotation_quaternion(cam2world)
    print("camera_rot", camera_rot, "camera_loc", camera.location)
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = default_sensor_width
    camera.data.lens = focal_in_mm
    camera.data.lens_unit = "MILLIMETERS"
    camera.data.shift_x = (intrinsic['W']/2 - intrinsic['u0'])/intrinsic['W']
    camera.data.shift_y = (intrinsics['H']/2 - intrinsics['v0'])/intrinsic['H']
    print(camera.data.lens, camera.data.shift_x, camera.data.shift_y)
    get_3x4_P_matrix_from_blender(camera)

def set_multi_camera_params(intrinsic, cam2world_array, render_opt,camera_obj_names=None):
    """
    Args:
        intrinsic_array: List[Dict]
            List of dictionaries, each containing {H: xx, W:xx, focal:xx, u0: xx, v0: xx}
            for each camera.
        cam2world_array: List[np.ndarray]
            List of 4x4 matrices representing the OpenCV coordinate system for each camera.
        camera_obj_names: List[str], optional
            List of names for the cameras. Default names will be "Camera_1", "Camera_2", etc.
    """
    num_cameras = len(intrinsic['focal_x'])
    
    if camera_obj_names is None:
        camera_obj_names = [f"Camera_{i+1}" for i in range(num_cameras)]
    
    for i in range(num_cameras):
        cam2world = cam2world_array[i]
        camera_obj_name = camera_obj_names[i]

         # Create or reuse the camera object
        if camera_obj_name not in bpy.data.objects:
            # Add a new camera
            new_camera = bpy.ops.object.camera_add(enter_editmode=True)
            camera = bpy.context.view_layer.objects.active
            camera.name = camera_obj_name
            print(f"Created new camera {camera_obj_name}")
            # Activate the camera
        else:
            camera = bpy.data.objects['Camera']
            
        bpy.context.scene.camera = camera
        
        
        # Set the camera parameters
        focal_in_mm = get_focal_in_mm(intrinsic['W'][i], intrinsic['focal_x'][i])
        bpy.context.scene.render.resolution_x = intrinsic['W'][i]
        bpy.context.scene.render.resolution_y = intrinsic['H'][i]
        camera.location = get_location(cam2world)
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = get_rotation_quaternion(cam2world)
        camera.data.sensor_fit = "HORIZONTAL"
        camera.data.sensor_width = default_sensor_width
        camera.data.sensor_height = (default_sensor_width*intrinsic['H'][i]*intrinsic['focal_x'][i])/(intrinsic['W'][i]*intrinsic['focal_y'][i])
        camera.data.lens = focal_in_mm
        camera.data.lens_unit = "MILLIMETERS"
        camera.data.shift_x = (intrinsic['W'][i]/2 - intrinsic['u0'][i])/intrinsic['W'][i]
        camera.data.shift_y = (intrinsic['v0'][i] - intrinsic['H'][i]/2)/(intrinsic['H'][i]*2)
        shift_y_json_path = f'{render_opt["output_dir"]}/info/{render_opt["road_seq"]}/shift_y.json'
        shift_x_json_path = f'{render_opt["output_dir"]}/info/{render_opt["road_seq"]}/shift_x.json'
        if os.path.exists(shift_y_json_path):
            with open(shift_y_json_path, 'r') as f:
                shift_y_data = json.load(f)
            shift_y = shift_y_data['shift_y']
            if shift_y[i] != -1:
                camera.data.shift_y = shift_y[i]
        if os.path.exists(shift_x_json_path):
            with open(shift_x_json_path, 'r') as f:
                shift_x_data = json.load(f)
            shift_x = shift_x_data['shift_x']
            if shift_x[i] != -1:
                camera.data.shift_x = shift_x[i]
        
        # print("==========================================================================")
        # print(camera.data.lens, camera.data.shift_x, camera.data.shift_y)
        # print(intrinsic['W'], intrinsic['H'], intrinsic['focal_x'][i], intrinsic['u0'][i], intrinsic['v0'][i])
        # get_3x4_P_matrix_from_blender(camera)
        # print(f"Setting camera {camera_obj_name} with location {camera.location}")
    
    