from common import *
from libcom import ImageHarmonizationModel,ShadowGenerationModel,color_transfer
import os
import cv2
import shutil,time
import argparse

task_name = 'image_harmonization'
def load_img(img_dir,mask_dir):
    image_files = os.listdir(img_dir)
    image_files.sort()
    mask_files = os.listdir(mask_dir)
    mask_files.sort()
    image_list, mask_list = [], []
    for i in range(len(image_files)):
        image_name,ext = os.path.splitext(image_files[i])
        mask_name,ext = os.path.splitext(mask_files[i])
        if image_name!= mask_name:
            print(f'image name: {image_name}, mask name: {mask_name}')
            print('Error: image and mask file names do not match!')
            return None
        img_path = os.path.join(img_dir, image_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])
        image_list.append(img_path)
        mask_list.append(mask_path)
    return image_list, mask_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim/result/standard_rcooper/standard_sim_16_rcooper_0.0_0.3/136-137-138-139/136-0/train/image', help='path to composite image')
    parser.add_argument('--mask_path', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim/result/standard_rcooper/standard_sim_16_rcooper_0.0_0.3/136-137-138-139_mask/136-0/train/image', help='path to composite mask')
    parser.add_argument('--result_dir', type=str, default='/home/ubuntu/duyuwen/RoCo-Sim/result/standard_rcooper/standard_sim_16_rcooper_0.0_0.3/image_harmonization', help='path to save result')
    parser.add_argument('--gpu_id', type=int, default=9, help='gpu id')
    args = parser.parse_args()
    img_path = args.img_path
    mask_path = args.mask_path

    result_dir = args.result_dir
   
    os.makedirs(result_dir, exist_ok=True)
    # device = torch.device(f"cuda:{args.gpu_id}")
    pct_model = ImageHarmonizationModel(model_type='PCTNet')
    print(f'begin testing {task_name}...')
    image_list, mask_list = load_img(img_path,mask_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for comp_img, comp_mask in zip(image_list, mask_list):
        begin_time = time.time()
        pct_res   = pct_model(comp_img, comp_mask)
        end_time = time.time()
        img_origin = cv2.imread(comp_img)
        img_name  = os.path.basename(comp_img)
        # grid_img = np.hstack([img_origin, pct_res])
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, pct_res)

        print('save result to ', res_path)
