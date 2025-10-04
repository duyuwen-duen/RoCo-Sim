import bpy
from blender_utils.camera.set_camera import set_multi_camera_params
from blender_utils.model.set_model import add_plane, add_model_params
from blender_utils.world.hdri import set_hdri
from blender_utils.render.render import render_scene, check_mkdir
from blender_utils.postprocess.compose import compose
from blender_utils.utils.common_utils import rm_all_in_blender

import numpy as np
import yaml
import sys
import os
import imageio.v2 as imageio
import shutil
import cv2
import time,re
from PIL import Image, ImageDraw


def check_state(render_opt):
    already_rendered = True
    for i in range(render_opt['cam2world'].shape[0]):
        print(f"================ camera :{i} ================")
        camera_name = f"Camera_{i+1}"
        cam_name,data_type = render_opt['save_path_list'][i].split('/')
        save_path = os.path.join(f"{render_opt['output_dir']}/{render_opt['setting_type']}",render_opt['road_seq'],cam_name,data_type,"image")
        
        if os.path.exists(save_path):
            image_list = os.listdir(save_path)
            image_name_list = []
            for image in image_list:
                image_name,ext = os.path.splitext(image)
                image_name_list.append(image_name)
            if render_opt['render_name'] not in image_name_list:
                already_rendered = False
                break
        else:
            already_rendered = False
            break   
            
    return already_rendered

def render(render_opt,gpu_id=0):
    if check_state(render_opt):
        print(f"Already rendered all cameras for {render_opt['render_name']}.........")
        return
    render_downsample = render_opt.get('render_downsample', 1)
    print(f"render_downsample: {render_downsample}")
    motion_blur_degree = render_opt.get('motion_blur_degree', 4)

    hdri_file = render_opt['hdri_file']
    intrinsic = render_opt['intrinsic']
    cam2world = render_opt['cam2world']
    background_RGB_path = render_opt['background_RGB'] 
    background_RGB_origin = [cv2.imread(path) for path in background_RGB_path]
    background_RGB = []
    for rgb in background_RGB_origin:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # [H,W,3]
        background_RGB.append(rgb)
    background_RGB = np.array(background_RGB)
    background_depth_path = render_opt['background_depth']
    print(background_depth_path[0])
    background_depth = np.array([cv2.imread(path, cv2.COLOR_BGR2GRAY).astype(np.float32)/100.0 for path in background_depth_path])
    background_depth = np.array([np.expand_dims(depth, axis=-1) for depth in background_depth])
    set_multi_camera_params(intrinsic, cam2world, render_opt)
    model_obj_names = []
    car_list = render_opt['cars']
    for car_obj in car_list:
        add_model_params(car_obj)
        add_plane(car_obj) # each car has a plane
        model_obj_names.append(car_obj['new_obj_name'])
        print(f"Added car {car_obj['new_obj_name']}")
   
    set_hdri(hdri_file, None)
   
    cache_dir = os.path.join(render_opt['output_dir'],render_opt['setting_type'],'cache',render_opt['road_seq'],render_opt['seq'])
    check_mkdir(os.path.join(cache_dir))
    # back up hdri and rgb image

    for i in range(render_opt['cam2world'].shape[0]):
        camera_name = f"Camera_{i+1}"
        render_output_backup_dir = os.path.join(cache_dir,camera_name, 'backup')
        check_mkdir(render_output_backup_dir)
        cam_name,data_type = render_opt['save_path_list'][i].split('/')
        save_path = os.path.join(f"{render_opt['output_dir']}/{render_opt['setting_type']}",render_opt['road_seq'],cam_name,data_type,"image")
        if os.path.exists(save_path):
            image_list = os.listdir(save_path)
            image_name_list = []
            for image in image_list:
                image_name,ext = os.path.splitext(image)
                image_name_list.append(image_name)
            if render_opt['render_name'] in image_name_list:
                print(f"Already rendered {render_opt['render_name']}.........")
                continue
        else:
            os.makedirs(save_path)
        imageio.imsave(os.path.join(render_output_backup_dir, render_opt['render_name'] + '_RGB.png'), background_RGB[i]) # save background RGB is necessary for rendering

        shutil.copy(hdri_file, os.path.join(render_output_backup_dir, render_opt['render_name'] + '_hdri.exr'))
        if isinstance(background_depth, np.ndarray):
            depth_and_occlusion = True
            background_depth_ = background_depth[i]
        else:
            depth_and_occlusion = False
            background_depth_ = background_depth
            
        render_scene(render_opt,intrinsic, model_obj_names, render_downsample, depth_and_occlusion,camera_name,gpu_id)
        compose(render_opt, background_RGB[i], background_depth_, render_downsample, motion_blur_degree, depth_and_occlusion,save_path,camera_name)
        print(f'Save image to {save_path}')
        
def generate_convex_hull(points):
    hull = ConvexHull(points)
    return points[hull.vertices]

def polygon_to_mask(polygon, height, width):
    # polygon = np.flip(polygon,axis=1)
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in polygon], outline=1, fill=1)
    mask = np.array(img)
    return mask


def accumulate_masks(mask_list, height, width):
    full_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in mask_list:
        mask = np.array(mask)
        # 累加遮罩
        full_mask[mask == 1] = 1
    return full_mask

def get_mask(render_opt,H,w,path,camera_id):

    mask = np.zeros((H,w))
    points_list = render_opt['mask']
    mask_list = []
    for car_obj in points_list:
        # 有1个点在画面中就消除
        points = np.array(car_obj['points'][camera_id]).reshape(8,2)
        # 扩展3dbbox
        center = np.mean(points, axis=0)
        vectors_to_center = points - center

        vectors_magnitude = np.linalg.norm(vectors_to_center, axis=1)

        expansion_factor = 1.3 
        expanded_vectors = vectors_to_center * expansion_factor

        expanded_points = center + expanded_vectors
        
        
        is_inimage = False
        for point in points:
            if point[0] >= 0 and point[0] < w and point[1] >= 0 and point[1] < H:
                is_inimage = True
                break
        
        if not is_inimage:
            continue
        
        polygon = generate_convex_hull(expanded_points)
        mask = polygon_to_mask(polygon.astype(int), H, w)
        mask_list.append(mask)
        
    full_mask = accumulate_masks(mask_list, H,w)
    mask_path = os.path.join(path, render_opt['render_name'].split('_')[0] + '_'+str(camera_id) + f'_mask.png')
    cv2.imwrite(mask_path,cv2.resize(full_mask, (1024,512))*255)

def inpainting_api(inpaint_input_path, inpaint_output_path,gpu_id=0):
    print(f"========== use gpu {gpu_id} to inpainting =============")
    current_dir = os.getcwd()
    os.chdir('/home/ubuntu/duyuwen/ChatSim/chatsim/background/inpainting/latent-diffusion')
    print(f"Current dir: {os.getcwd()}")
    os.system(
        f"python scripts/inpaint.py --indir {inpaint_input_path} --outdir {inpaint_output_path} --gpu {gpu_id}"
    )
    os.chdir(current_dir)

def inpainting(render_opt,H,W,rgb,gpu_id=0):
    inpainting_begin_time = time.time()
    path = os.path.dirname(render_opt['scene_file']).replace('blender_npz','cache')
    input_path = os.path.join(path, 'input')
    output_path = os.path.join(path, 'output')
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    file_list = sorted(os.listdir(output_path),key = lambda x:int(x.split('.')[0]))
    image_list = []
    if len(render_opt['mask']) > 0:
        if len(file_list) == 0 or int(render_opt['render_name'].split('_')[0]) > int(file_list[-1].split('_')[0]):
            print(f"Inpainting {render_opt['render_name']}.........")
            os.system(
                f"rm -r {os.path.join(path, 'input')}/*"
            )
            # os.system(
            #     f"rm -r {os.path.join(path, 'output')}/*"
            # )
            for i in range(render_opt['cam2world'].shape[0]):
                get_mask(render_opt,H,W,input_path,i)
                cv2.imwrite(os.path.join(input_path, render_opt['render_name'].split('_')[0] + '_'+str(i) + '.png'), cv2.resize(rgb[i], (1024,512)))
            
            inpainting_api(input_path,output_path,gpu_id)
            
        else :
            print(f"Already inpainted {render_opt['render_name']}.........")
        
        for i in range(render_opt['cam2world'].shape[0]):
            image = cv2.imread(os.path.join(output_path, render_opt['render_name'].split('_')[0] + '_'+ str(i) + '.png'))
            image_list.append(cv2.resize(image, (W,H)))
    else:
        for i in range(render_opt['cam2world'].shape[0]):
            image_list.append(rgb[i])  

    inpainting_end_time = time.time()
    print(f"Inpainting time: {inpainting_end_time - inpainting_begin_time}s")
    return image_list
        
def main():
    def parse_argv(argv):
        result = []
        for i in range(1, len(argv)):
            # When encountering "--", take the next element as the value.
            if argv[i] == '--':
                if i + 1 < len(argv):
                    result.append(argv[i + 1])
        return result
    
    argv = sys.argv
    argv = parse_argv(argv)

    render_yaml = argv[0]

    start_frame = int(argv[1])
    end_frame = int(argv[2])
    gpu_id = int(argv[3])
    
    for frame_id, yaml_file in enumerate(sorted(os.listdir(render_yaml),key = lambda x: [int(num) if num else '' for num in re.findall(r'\d+', x)])):
        if frame_id < start_frame or frame_id > end_frame:
            continue
        begin_time = time.time()
        frame = os.path.splitext(yaml_file)[0]
        print(f"Rendering frame {frame}.........")
        
        with open(os.path.join(render_yaml,yaml_file), 'r') as file:
            render_opt = yaml.safe_load(file)
        
        bpy.ops.wm.read_homefile(app_template="")
        rm_all_in_blender()

        # read config from scene
        scene_data = render_opt['scene_file']
        data_dict = np.load(scene_data)

        H = data_dict['H'].tolist()
        W = data_dict['W'].tolist()     
        v0 = data_dict['v0'].tolist()
        u0 = data_dict['u0'].tolist()
        focal_x = data_dict['focal_x'].tolist()
        focal_y = data_dict['focal_y'].tolist()
        render_opt['intrinsic'] = {"H":H, "W":W, "v0":v0, "u0":u0,"focal_x":focal_x,"focal_y":focal_y}
        render_opt['cam2world'] = data_dict['extrinsic']
        render_opt['save_path_list'] = data_dict['save_path'].tolist()
        
        
        ############################ inpainting ###############################
        # image_list = inpainting(render_opt,H,W,data_dict['rgb'],gpu_id)

        ############################ rendering ###############################
        render_opt['background_RGB'] = data_dict['rgb']
        render_opt['background_depth'] = data_dict['depth']
        render_opt['names'] = data_dict['names']
        render_opt['seq'] = f"{start_frame}_{end_frame}"

        render(render_opt,gpu_id)
        end_time = time.time()
        print(f"Frame {frame} total rendering time: {end_time - begin_time}s")

if __name__=="__main__":
    main()