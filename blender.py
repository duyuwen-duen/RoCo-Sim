import os 
import subprocess
import json
import time
import numpy as np
import argparse
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'roco_sim/foreground'))


def setup_paths(args):
    """
    Create and return necessary paths based on input arguments.
    """
    log_dir = f'{args.home_dir}/result/{args.dataset_type}/{args.setting_type}/cache/{args.name}/log'
    yaml_dir = f'{args.home_dir}/result/{args.dataset_type}/{args.setting_type}/yaml/{args.name}/{args.data_type}'
    blender_path = f'{args.home_dir}/roco_sim/foreground/Blender/blender-3.5.1-linux-x64/blender'
    blender_utils = f"{args.home_dir}/roco_sim/foreground/Blender/utils"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, yaml_dir, blender_path, blender_utils

def calculate_blender_tasks(yaml_dir, gpu_list, each_gpu_blender_num):
    """
    Calculate and return number of frames per GPU and associated indices.
    """
    yaml_num = len(os.listdir(yaml_dir))
    gpu_num = len(gpu_list)
    total_blender_processes = gpu_num * each_gpu_blender_num
    frames_per_blender = max(1, yaml_num // total_blender_processes)
    return yaml_num, gpu_num, total_blender_processes, frames_per_blender

def create_subprocesses(blender_path, blender_utils, yaml_dir, log_dir, gpu_list, total_blender_processes, frames_per_blender,yaml_num):
    """
    Create subprocesses for running Blender in parallel.
    """
    processes = []
    gpu_indices = [int(gpu) for gpu in gpu_list]
    for i in range(total_blender_processes):
        start_frame = i * frames_per_blender
        end_frame = min((i + 1) * frames_per_blender - 1, yaml_num - 1)
        if end_frame < start_frame:  # No frames left
            continue
        use_gpu = gpu_indices[i % len(gpu_indices)]
        log_file = os.path.join(log_dir, f"{start_frame}_{end_frame}.txt")
        command = (
            f"{blender_path} -b --python {os.path.join(blender_utils, 'blender_utils/main_multicar.py')} "
            f"-- {yaml_dir} -- {start_frame} -- {end_frame} -- {use_gpu} > {log_file} 2>&1"
        )
        print(f"Starting process {i}, frames {start_frame} to {end_frame}, using GPU-{use_gpu}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((process, start_frame, end_frame))
    return processes

def monitor_processes(processes, finish_list):
    """
    Monitor the subprocesses for completion and handle results.
    """
    flag = np.zeros(len(processes))
    for idx, (process, start_frame, end_frame) in enumerate(processes):
        stdout, stderr = process.communicate()
        seq_name = f"{start_frame}_{end_frame}"
        if process.returncode != 0:
            print(f"Error: {seq_name} failed with return code {process.returncode}")
            print(f"Standard Error (process {process.pid}): {stderr.decode()}")
            flag[idx] = 0
        else:
            print(f"{seq_name} finished successfully")
            if seq_name not in finish_list:
                finish_list.append(seq_name)
            flag[idx] = 1
    return flag

def blender(args):
    '''
    Model the camera and cars, then blender the foreground to background
    Blender it with multiple gpus parallelly
    '''
    print(f"\033[1;32m{'#'*30} blender start {'#'*30}\033[0m")
    setting_type = args.setting_type
    name = args.name
    log_dir, yaml_dir, blender_path, blender_utils = setup_paths(args)

    # Check available YAMLs and GPUs
    yaml_num, gpu_num, total_blender_processes, frames_per_blender = calculate_blender_tasks(
        yaml_dir, args.gpu_list.split(','), args.each_gpu_blender_num
    )
    print(f"{'='*30} yaml_num: {yaml_num}, gpu_num: {gpu_num} {'='*30}")

    finish_list = []
    while len(finish_list) < total_blender_processes:
        start_time = time.time()

        # Create and monitor processes
        processes = create_subprocesses(
            blender_path, blender_utils, yaml_dir, log_dir, args.gpu_list.split(','), total_blender_processes, frames_per_blender,yaml_num
        )
        flag = monitor_processes(processes, finish_list)

        # Check if all processes are complete
        end_time = time.time()
        print(f"Batch completed in {end_time - start_time:.2f} seconds")
        if np.sum(flag) == len(flag):
            break

    print(f"\033[1;32m{'#' * 30} Blender Complete {'#' * 30}\033[0m")

def harmonize(args):
    '''
    harmonize the foreground and background
    '''
    print(f"\033[1;32m{'#'*30} harmonize start {'#'*30}\033[0m")
    data_dir = f'{args.home_dir}/result/{args.dataset_type}/{args.setting_type}/{args.name}'
    cam_list = os.listdir(data_dir)
    cam_list = [cam for cam in cam_list if cam != 'coop']
    for cam in cam_list:
        img_path = f'{data_dir}/{cam}/{args.data_type}/image'
        mask_path = f'{data_dir}_mask/{cam}/{args.data_type}/image'
        result_dir = f'{data_dir}/{cam}/{args.data_type}/image'
        command = f"cd {os.path.join(args.home_dir, 'roco_sim/foreground/libcom/tests')}"
        command += f" && python image_harmonization.py --img_path {img_path} --mask_path {mask_path} --result_dir {result_dir}"
        os.system(command)

def style_transfer(args):
    '''
    style transfer the foreground and background
    '''
    print(f"\033[1;32m{'#'*30} style transfer start {'#'*30}\033[0m")
    data_dir = f'{args.home_dir}/result/{args.dataset_type}/{args.setting_type}/{args.name}'
    cam_list = os.listdir(data_dir)
    cam_list = [cam for cam in cam_list if cam != 'coop']
    for cam in cam_list:
        img_path = f'{data_dir}/{cam}/{args.data_type}/image'
        mask_path = f'{data_dir}/{cam}_mask/{args.data_type}/image'
        result_dir = f'{data_dir}/{cam}/{args.data_type}/image'
        image_list = os.listdir(img_path)
        for image in image_list:
            image_path = f'{img_path}/{image}'
            command = f'cd {os.path.join(args.home_dir, "roco_sim/postprocess/img2img-turbo")}'
            command += f' && python src/inference_unpaired.py --model_name {args.style_transfer} \
            --input_image {image_path} --output_dir {result_dir}'
            os.system(command)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dir', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim', help='home directory')
    parser.add_argument('--setting_type', type=str, default='16_read-pos_0.3_0.6', help='setting type')
    parser.add_argument('--name', type=str, default='136-137-138-139', help='name of the sequence')
    parser.add_argument('--gpu_list', type=str, default='2,3,4', help='gpu list')
    parser.add_argument('--each_gpu_blender_num', type=int, default=1, help='each blender num') 
    parser.add_argument('--dataset_type', type=str, default='rcooper', help='dataset type')
    parser.add_argument('--data_type', type=str, default='train', help='data type')
    parser.add_argument('--harmonize', type=bool, default=True, help='harmonize')
    parser.add_argument('--style_transfer', type=str, default='', help='style transfer')
    args = parser.parse_args()
    blender(args)
    if args.harmonize:  
        harmonize(args)
    if args.style_transfer != '':
        style_transfer(args)
    
    
